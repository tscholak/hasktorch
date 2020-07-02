{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Prelude hiding ( tanh )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           GHC.Generics
import           GHC.TypeLits

import Torch.Typed
import           Torch.Data.Pipeline
import Torch (ATenTensor)
import Torch.Internal.Class (Castable)
import Control.Monad.IO.Class (MonadIO)


--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------

data MLPSpec (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
             (dtype :: DType)
             (device :: (DeviceType, Nat))
  = MLPSpec

data MLP (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
         (dtype :: DType)
         (device :: (DeviceType, Nat))
  = MLP { layer0 :: Linear inputFeatures  hiddenFeatures dtype device
        , layer1 :: Linear hiddenFeatures hiddenFeatures dtype device
        , layer2 :: Linear hiddenFeatures outputFeatures dtype device
        } deriving (Show, Generic)

instance
  (StandardFloatingPointDTypeValidation device dtype) => HasForward
    (MLP inputFeatures outputFeatures hiddenFeatures dtype device)
    (Tensor device dtype '[batchSize, inputFeatures])
    (Tensor device dtype '[batchSize, outputFeatures])
 where
  forward MLP {..} = forward layer2 . tanh . forward layer1 . tanh . forward layer0

instance
  ( KnownDevice device
  , KnownDType dtype
  , All KnownNat '[inputFeatures, outputFeatures, hiddenFeatures]
  , RandDTypeIsValid device dtype
  ) => Randomizable
    (MLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (MLP     inputFeatures outputFeatures hiddenFeatures dtype device)
 where
  sample MLPSpec =
    MLP <$> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec

xor
  :: forall batchSize dtype device
   . KnownDevice device
  => Tensor device dtype '[batchSize, 2]
  -> Tensor device dtype '[batchSize]
xor t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
 where
  a = select @1 @0 t
  b = select @1 @1 t

newtype Xor = Xor { iters :: Int }

instance ( KnownNat batchSize
         , KnownDevice device
         , KnownDType dtype
         , RandDTypeIsValid device dtype
         , ComparisonDTypeIsValid device dtype
         ) => Dataset IO Xor (Tensor device dtype '[batchSize, 2]) where
  getBatch _ _ =  toDType @dtype @'Bool .
                  gt (toDevice @device (0.5 :: CPUTensor dtype '[]))
                  <$> rand @'[batchSize, 2] @dtype @device
  numIters = iters

-- this should be part of hasktorch, maybe in a new Torch.Typed.Trainer module
-- there needs to be another one for the untyped API in Torch.Trainer
class
  ( 'Just device ~ GetDevice model
  , 'Just device ~ GetDevice batch
  , 'Just dtype ~ GetDType model
  , 'Just dtype ~ GetDType batch
  ) => HasTrainingStep model batch device dtype where
  trainingStep :: model -> (batch, Int) -> Loss device dtype

-- this is similar to torch lightning's training_step
instance
  ( StandardFloatingPointDTypeValidation device dtype
  , SqueezeAll '[batchSize, 1] ~ '[batchSize]
  , KnownDevice device
  ) => HasTrainingStep (MLP 2 1 4 dtype device) (Tensor device dtype '[batchSize, 2]) device dtype where
  trainingStep model (batch, iter) = 
    let actualOutput   = squeezeAll . ((sigmoid .) . forward) model $ batch
        expectedOutput = xor batch
    in  mseLoss @ReduceMean actualOutput expectedOutput

-- this or similar should be a library function
train
  :: forall batch model optim device dtype dataset parameters gradients tensors
   . ( Dataset IO dataset batch
     , HasTrainingStep model batch device dtype
     , Parameterized model parameters
     , HasGrad (HList parameters) (HList gradients)
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , Castable (HList gradients) [ATenTensor]
     , Optimizer optim gradients tensors dtype device
     , HMapM' IO MakeIndependent tensors parameters
     )
  => LearningRate device dtype
  -> dataset
  -> model
  -> optim
  -> IO (model, optim)
train learningRate dataset model optim = do
  fold <- makeFold dataset
  fold $ FoldM
    ( \(model, optim) (batch, iter) -> do
        let loss = trainingStep @model @batch model (batch, iter)
        when (iter `mod` 2500 == 0) (print loss)
        runStep model optim loss learningRate
    )
    (pure (model, optim))
    pure

type Device_ = '( 'CPU, 0)
type DType_ = 'Float

main :: IO ()
main = do
  let numIters = 100000
      learningRate = 0.1
  model <- sample (MLPSpec :: MLPSpec 2 1 4 DType_ Device_)
  let optim = mkAdam 0 0.9 0.999 (flattenParameters model)
  (model', _) <- train @(Tensor Device_ DType_ '[256, 2]) learningRate (Xor { iters = numIters }) model optim
  print model'
