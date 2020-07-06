{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
module Torch.Data.Pipeline ( takeBatch
                           , readBatches
                           , readBatchesConcurrently
                           , makeFold'
                           , makeFoldWithTransform'
                           , makeConcurrentFold'
                           , makeFold
                           , makeFoldWithTransform
                           , makeConcurrentFold
                           , foldFromProducer
                           , L.FoldM(..)
                           , Dataset(..)
                           , DatasetMock(..)
                           , ConcurrentDataset(..)
                           ) where

import           Control.Concurrent.Async.Lifted
import qualified Control.Foldl as L
import           Control.Monad
import           Data.Maybe (isJust)
import           Pipes
import           Pipes.Concurrent
import qualified Pipes.Prelude as P

import           Control.Monad.Base              (MonadBase, liftBase)
import           Control.Monad.Trans.Control (control, MonadBaseControl(..))
import Control.Monad.Trans.Free (wrap, iterT, FreeF(..), runFreeT, FreeT(..))
import Control.Exception (finally, bracket)


foo :: (Monad m) => FreeT ((,) a) m r -> Producer a m r
foo f = do
  x <- lift (runFreeT f)
  case x of
    Pure r -> pure r
    Free (a, f') -> yield a >> foo f'

bar :: (Monad m) => Producer a m r -> FreeT ((,) a) m r
bar p = do 
  x <- lift (next p)
  case x of
    Left r -> pure r
    Right (a, p') -> wrap (a, bar p')

baz
  :: forall m a b r
   . (MonadBaseControl IO m, MonadBase IO m)
  => Buffer b -- buffer
  -> (a -> ListT m b) -- create a dataset by seeding it with an `a`
  -> ListT m a -- an effectful iteration of `a`s
  -> m (ListT m b)
baz buffer f p =
  let l output = let step = flip $ mappend . Concurrently . runEffect . (>-> toOutput' output) . enumerate . f
                 in  join . P.fold step mempty runConcurrently $ enumerate p
      r = pure . fromInput'
  in  Select . snd <$> withBuffer' buffer l r

withBuffer'
    :: (MonadBaseControl IO m, MonadBase IO m)
    => Buffer a
    -> (Output a -> m l)
    -> (Input  a -> m r)
    -> m (l, r)
withBuffer' buffer fOutput fInput = liftedBracket
  (liftBase $ spawn' buffer)
  (\(_, _, seal) -> liftBase $ atomically seal)
  (\(output, input, seal) ->
    concurrently
      (fOutput output `liftedFinally` (liftBase $ atomically seal))
      (fInput  input  `liftedFinally` (liftBase $ atomically seal))
  )

fromInput' :: (MonadBase IO m) => Input a -> Producer' a m ()
fromInput' input = loop
  where
    loop = do
        ma <- liftBase $ atomically $ recv input
        case ma of
            Nothing -> return ()
            Just a  -> do
                yield a
                loop

toOutput' :: (MonadBase IO m) => Output a -> Consumer' a m ()
toOutput' output = loop
  where
    loop = do
        a     <- await
        alive <- liftBase $ atomically $ send output a
        when alive loop

liftedBracket :: MonadBaseControl IO m => m a -> (a -> m b) -> (a -> m c) -> m c
liftedBracket acquire release action = control $ \runInIO ->
    bracket (runInIO acquire)
            (\saved -> runInIO (restoreM saved >>= release))
            (\saved -> runInIO (restoreM saved >>= action))

liftedFinally :: MonadBaseControl IO m => m a -> m b -> m a
liftedFinally a sequel = control $ \runInIO ->
                           finally (runInIO a)
                                   (runInIO sequel)

instance (MonadBase IO m) => MonadBase IO (Proxy a' a b' b m) where
  liftBase = lift . liftBase

type Iter = Int
type WorkerId = Int

data DatasetMock m tensor = DatasetMock { getBatchMock :: Int -> m tensor
                                        , numItersMock :: Iter
                                        }

class (MonadPlus m, MonadIO m, MonadBaseControl IO m) => Dataset m dataset batch  where
  getBatch :: dataset -> Iter -> m batch
  numIters :: dataset -> Int

class Dataset m dataset batch => ConcurrentDataset m dataset batch where
  getBatchConcurrently :: WorkerId -> dataset -> Iter -> m batch

takeBatch :: MonadIO m => Input (Maybe batch) -> Producer batch m ()  
takeBatch input = fromInput input >-> yieldMore
  where yieldMore = forever $ await >>= \case
          Just batch -> yield batch
          Nothing -> return ()

readBatches'
  :: MonadIO m
  => Int
  -> (Int -> m a)
  -> Output (Maybe (a, Int))
  -> Effect m ()
readBatches' numIters getBatch outputBox = 
  for (each [1..numIters]) (\iter -> yieldBatch iter >-> toOutput outputBox)
    where yieldBatch iter = (lift . runBatch) iter >>= yield
          runBatch iter = if numIters == iter then pure Nothing else (\batch -> Just (batch, iter)) <$> getBatch iter

readBatchesConcurrently :: forall dataset batch m .
  (ConcurrentDataset m dataset batch) => Int -> dataset -> Output (Maybe (batch, Int))  -> Effect m () 
readBatchesConcurrently workerId dataset outputBox = 
  readBatches' (numIters @m @dataset @batch dataset + 1) (getBatchConcurrently workerId dataset) outputBox

readBatches :: forall dataset batch m.
  (Dataset m dataset batch) => dataset -> Output (Maybe (batch, Int))  -> Effect m () 
readBatches dataset outputBox = readBatches' (numIters @m @dataset @batch dataset + 1) (getBatch dataset) outputBox

runTransforms :: MonadIO m => (batch -> batch') -> Input (Maybe (batch, Int)) -> Output (Maybe (batch', Int)) -> Effect m ()
runTransforms transforms transformBox trainBox = fromInput transformBox >-> P.map (fmap (\(batch, iter) -> (transforms batch, iter))) >-> toOutput trainBox

makeFold' :: (Dataset m2 dataset batch, MonadIO m, MonadIO m2)
    => dataset
    -> m2 (L.FoldM m (batch, Int) b -> m b, Async (StM m2 ()))
makeFold' dataset = do
  (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
  batchThread <- async $ void $ runEffect $ readBatches dataset toBatches
  pure (foldFromProducer (takeBatch fromBatches), batchThread)

makeConcurrentFold' :: (MonadIO m2, ConcurrentDataset m2 dataset batch', MonadIO m)
  => (batch' -> batch)
  -> dataset
  -> Int
  -> m2 (L.FoldM m (batch, Int) b -> m b, [Async (StM m2 ())])
makeConcurrentFold' transforms dataset numWorkers = do
  -- Buffer size is equal to numWorkers so that each thread can yield a batch.
  -- This is not actually the enforced behaviour, one thread may fill the buffer with multiple batches,
  -- but it should be better than a buffer size of 1 in this multithreaded case.
  (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded numWorkers)
  (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded numWorkers)
  batchThreads <- forM [1..numWorkers] $ \workerId -> async $ void $ runEffect $ readBatchesConcurrently workerId dataset toTransformBox
  async $ runEffect $ runTransforms transforms fromTransformBox toBatches
  pure  $ (foldFromProducer (takeBatch fromBatches), batchThreads)

makeFoldWithTransform' :: (MonadIO m, MonadIO m2, Dataset m2 dataset batch)  
  => (batch -> batch')
  -> dataset
  -> m2 (L.FoldM m (batch', Int) b -> m b, Async (StM m2 ()))
makeFoldWithTransform' transforms dataset = do
          -- TODO: we can allow different buffer sizes
          -- which would be necessary for data echoing
            (toTransformBox, fromTransformBox, sealTransform) <- liftIO $ spawn' (bounded 1)
            (toBatches, fromBatches, sealBatch) <- liftIO $ spawn' (bounded 1)
            batchThread <- async $ void $ runEffect $ forever $ readBatches dataset toTransformBox 
            async $ runEffect $ runTransforms transforms fromTransformBox toBatches
            pure $ (foldFromProducer (takeBatch fromBatches), batchThread)

makeFold :: (Dataset m2 dataset batch, MonadIO m, MonadIO m2)
    => dataset
    -> m2 (L.FoldM m (batch, Int) b -> m b)
makeFold = fmap fst . makeFold' 

makeFoldWithTransform :: (MonadIO m, MonadIO m2, Dataset m2 dataset batch)  
  => (batch -> batch')
  -> dataset
  -> m2 (L.FoldM m (batch', Int) b -> m b)
makeFoldWithTransform transf = fmap fst . makeFoldWithTransform' transf 

makeConcurrentFold :: (MonadIO m2, MonadIO m, ConcurrentDataset m2 dataset batch')
  => (batch' -> batch)
  -> dataset
  -> Int
  -> m2 (L.FoldM m (batch, Int) b -> m b)
makeConcurrentFold transforms dataset = fmap fst . makeConcurrentFold' transforms dataset
  
foldFromProducer :: Monad m => Producer a m () -> L.FoldM m a b -> m b
foldFromProducer prod fold = (L.impurely P.foldM) fold prod
