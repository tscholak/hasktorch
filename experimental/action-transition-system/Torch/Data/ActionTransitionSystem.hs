{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Torch.Data.ActionTransitionSystem where

import Prelude hiding (lookup)
import GHC.Generics
import Control.Lens
import Data.Generics.Product
import Data.Generics.Sum
import Data.Kind (Type)
import Control.Foldl (Fold(..), fold, head)
import Control.Applicative (liftA2, pure, Alternative(..), empty, (<|>))
import Control.Monad (mfilter, MonadPlus(..))
import Data.Text (pack, Text)
import GHC.IO (unsafePerformIO)
-- import qualified Control.Monad.Gen as Fresh (GenT(..), runGen, runGenT, gen, Gen)
import Control.Monad.Yoctoparsec (Parser)
import Control.Monad.Trans.Maybe (MaybeT(..))
import Control.Monad.Trans.Free (wrap, iterTM, runFreeT, Free, FreeT(..), FreeF(..))
import Control.Monad.State (evalStateT, MonadState(..), StateT (..), runStateT, get, put, modify)
import Control.Monad (ap, void)
import Control.Monad.Trans (MonadTrans(lift))
import Data.Map as Map (elems, toList, (!), adjust, update, keys, null, insertWith, singleton, fromList, unionWith, Map, insert, lookup)
import Data.Set as Set (filter, cartesianProduct, unions, toList, fromList, member, singleton, union, Set, insert, findIndex)
import qualified Data.Set as Set (empty)
import qualified Data.Map as Map (empty)
import Data.List as List (filter, sort, nub)
import Control.Monad.Reader (MonadReader(..), ask, local, runReaderT, ReaderT)
import Hedgehog (distributeT, discover, checkParallel, PropertyT, check, Property, (===), forAll, property, Gen, MonadGen(..))
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import Control.Monad (guard)
import Data.Maybe (fromMaybe)
import Control.Monad (join)
import Bound (fromScope, toScope, instantiate1, abstract1, (>>>=), Scope)
import Data.Functor.Classes (Eq1(..), Ord1(..), Show1(..))
import Data.Deriving (deriveEq1, deriveOrd1, deriveShow1)
import Bound (Var)
import Data.Functor.Classes (eq1)
import Data.Functor.Classes (compare1)
import Data.Functor.Classes (showsPrec1)
import Control.Monad.Morph (MFunctor, hoist)
import Hedgehog.Internal.Distributive (Transformer, MonadTransDistributive)
-- import Control.Monad.Gen (Successor)
import Control.Monad.Writer.Class (MonadWriter(..))
import Control.Monad.IO.Class (MonadIO(..))
import Control.Monad.Cont.Class (MonadCont(..))
import Control.Monad.Error.Class (MonadError(..))
import Control.Monad.Fix (MonadFix(..))
import Control.Monad.Trans.Identity (IdentityT)
import Control.Monad.Trans.Cont (ContT)
import Control.Monad.Trans.RWS (RWST)
import Control.Monad.Trans.Except (ExceptT)
import Control.Monad.Trans.Writer (WriterT)
import qualified Control.Monad.Writer.Strict as SW
import qualified Control.Monad.State.Strict as SS
import Control.Monad.Trans.Reader (asks)

-- https://stackoverflow.com/questions/17675054/deserialising-with-ghc-generics?rq=1
-- http://hackage.haskell.org/package/cereal-0.5.8.1/docs/Data-Serialize.html
-- https://hackage.haskell.org/package/attoparsec-0.13.2.4/docs/src/Data.Attoparsec.Text.Internal.html
-- https://hackage.haskell.org/package/yoctoparsec-0.1.0.0/docs/src/Control-Monad-Yoctoparsec.html#Parser
-- https://vaibhavsagar.com/blog/2018/02/04/revisiting-monadic-parsing-haskell/
-- https://github.com/alphaHeavy/protobuf/blob/46cda829cf1e7b6bba2ff450e267fb1a9ace4fb3/src/Data/ProtocolBuffers/Ppr.hs

data Env = Env
  { pos :: Pos
  , meta :: Maybe M
  , rEnv :: REnv
  , aEnv :: AEnv
  -- , validActionsMask :: Map Pos (Set Action)
  } deriving (Eq, Ord, Show, Generic)

defaultEnv :: Env
defaultEnv = Env
  { pos = Pos 0
  , meta = Nothing
  , rEnv = REnv
    { parentPos = Nothing
    , parents = mempty
    , relations = mempty
    }
  , aEnv = AEnv
    { currentScope = Nothing
    , knownScopes = mempty
    , attentionMask = mempty
    , keyPaddingMask = mempty
    }
  -- , validActionsMask = mempty
  }

newtype Pos = Pos { unPos :: Int }
  deriving (Eq, Ord, Show, Num, Generic)

data Relation =
    ChildParentRelation
  | ParentChildRelation
  | SiblingDistRelation { siblingDist :: Int }
  deriving (Eq, Ord, Show, Generic)

data REnv = REnv
  { parentPos :: Maybe Pos
  , parents :: Map Pos (Set Pos)
  , relations :: Map (Pos, Pos) (Set Relation)
  } deriving (Eq, Ord, Show, Generic)

type ScopeId = Text

data AttentionScope = AttentionScope
  { scopeKind :: AttentionKind
  , scopeConnections :: Set ScopeId
  , scopePositions :: Set Pos
  } deriving (Eq, Ord, Show, Generic)

data AttentionKind = 
    BidirectionalAttention
  | BackwardAttention
  | ForwardAttention
    deriving (Eq, Ord, Show, Generic)

data AEnv = AEnv
  { currentScope :: Maybe ScopeId
  , knownScopes :: Map ScopeId AttentionScope
  , attentionMask :: Set (Pos, Pos)
  , keyPaddingMask :: Set Pos
  } deriving (Eq, Ord, Show, Generic)

data M = D Text | C Text | S Text
  deriving (Eq, Ord, Show, Generic)

data Action =
    L
  | R
  | Grow
  | Reduce
  | IToken Int
  | SToken Text
  | BToken Bool
    deriving (Eq, Ord, Show)

type To t a = a -> t Action
type From b a = Parser (StateT Env b) Action a

choice :: Alternative f => [f a] -> f a
choice = foldr (<|>) empty

option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan
  where scan = (end *> pure []) <|> liftA2 (:) p scan

skipMany :: Alternative f => f a -> f ()
skipMany p = scan
  where scan = (p *> scan) <|> pure ()

skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p

ancestralRelations :: forall f . Monad f => Pos -> StateT REnv f ()
ancestralRelations pos = get >>= (go . view (field @"parentPos"))
 where go Nothing          = pure ()
       go (Just parentPos) = 
         let rels' = update (Map.singleton (pos, parentPos) . Set.singleton $ ChildParentRelation)
                            (Map.singleton (parentPos, pos) . Set.singleton $ ParentChildRelation)
         in  modify (field @"relations" %~ (update rels'))
       update rels' rels = Map.unionWith Set.union rels' rels

siblingRelations :: forall f . Monad f => Pos -> StateT REnv f ()
siblingRelations pos = get >>= ap (go . view (field @"parentPos")) (view (field @"parents"))
  where go Nothing          parents = pure ()
        go (Just parentPos) parents = do
          let siblings = maybe mempty (Set.insert pos) $ lookup parentPos parents
              sibIndex = findIndex pos siblings
              step pos' (rels', idx) =
                let rels'' = update (Map.singleton (pos, pos') . Set.singleton . SiblingDistRelation $ sibIndex - idx)
                                    (Map.singleton (pos', pos) . Set.singleton . SiblingDistRelation $ idx - sibIndex)
                in  (update rels'' rels', idx + 1)
              (rels, _) = foldr step (mempty, 0) siblings
          modify (field @"relations" %~ (update rels))
          modify (field @"parents" %~ (Map.insert parentPos siblings))
        update rels' rels = Map.unionWith Set.union rels' rels

updateRelations :: forall f . Monad f => Pos -> StateT REnv f ()
updateRelations = ancestralRelations @f >> siblingRelations @f

updateAttention :: forall f. Monad f => Pos -> StateT AEnv f ()
updateAttention pos = do
  mScopeId <- (^. field @"currentScope") <$> get
  knownScopes <- (^. field @"knownScopes") <$> get
  case mScopeId of
    Just scopeId -> do
      modify (field @"attentionMask" %~ go pos scopeId knownScopes)
      modify (field @"knownScopes" %~ go' pos scopeId)
    Nothing -> pure ()
  modify (field @"keyPaddingMask" %~ Set.insert pos)
  where
    go pos thisScopeId knownScopes mask =
      let constrainAttention BidirectionalAttention = id
          constrainAttention BackwardAttention = Set.filter (uncurry (>=))
          constrainAttention ForwardAttention = Set.filter (uncurry (<=))
          mkMask kind from to = constrainAttention kind $ Set.cartesianProduct from to
          thisScope = knownScopes ! thisScopeId
          attendSelf = Set.singleton pos
          attendTo = Set.unions $ scopePositions . (knownScopes !) <$> Set.toList (scopeConnections $ thisScope)
          attendFrom = List.filter (member thisScopeId . scopeConnections) $ Map.elems knownScopes
          mask' =
            Set.unions
              [ mkMask (scopeKind thisScope) attendSelf attendTo
              , Set.unions $ (\thatScope -> mkMask (scopeKind thatScope) (scopePositions thatScope) attendSelf) <$> attendFrom
              , mkMask (scopeKind thisScope) attendSelf attendSelf
              ]
       in Set.union mask' mask
    go' pos = Map.adjust (field @"scopePositions" %~ (Set.insert pos))

-- TODO: move state updates somewhere else?
token :: forall b t . Monad b => Parser (StateT Env b) t t
token = do
  t <- wrap $ FreeT . pure . Pure
  pos <- (^. field @"pos") <$> get
  zoom (field @"rEnv") . lift . updateRelations $ pos
  zoom (field @"aEnv") . lift . updateAttention $ pos
  modify (field @"pos" %~ (+1))
  pure t

is :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
is t = mfilter (== t) token

isNot :: (MonadPlus b, Eq t) => t -> Parser (StateT Env b) t t
isNot t = mfilter (/= t) token

class ToActions (t :: Type -> Type) (a :: Type) where
  toActions :: To t a

  default toActions :: (Generic a, GToActions t (Rep a)) => To t a
  toActions = gToActions @t . GHC.Generics.from

class FromActions (b :: Type -> Type) (a :: Type) where
  fromActions :: From b a

  default fromActions :: (Monad b, Generic a, GFromActions b (Rep a)) => From b a
  fromActions = GHC.Generics.to <$> gFromActions @b

class GToActions (t :: Type -> Type) (f :: Type -> Type) where
  gToActions :: forall a . To t (f a)

class GFromActions (b :: Type -> Type) (f :: Type -> Type) where
  gFromActions :: forall a . From b (f a)

instance GToActions t f => GToActions t (M1 i c f) where
  gToActions = gToActions @t . unM1

instance (Monad b, GFromActions b f, Datatype d) => GFromActions b (D1 d f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . D . pack . datatypeName @d $ undefined)
    pos <- (^. field @"pos") <$> get
    modify $ field @"rEnv" . field @"parentPos" .~ (pure pos)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Constructor c) => GFromActions b (C1 c f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . C . pack . conName @c $ undefined)
    M1 <$> gFromActions @b

instance (Monad b, GFromActions b f, Selector s) => GFromActions b (S1 s f) where
  gFromActions = do
    modify $ field @"meta" .~ (pure . S . pack . selName @s $ undefined)
    M1 <$> gFromActions @b

instance ToActions t a => GToActions t (K1 i a) where
  gToActions = toActions @t . unK1

instance (Monad b, FromActions b a) => GFromActions b (K1 i a) where
  gFromActions = K1 <$> fromActions @b

instance Alternative t => GToActions t U1 where
  gToActions _ = empty

instance Monad b => GFromActions b U1 where
  gFromActions = pure U1

instance GToActions t V1 where
  gToActions v = v `seq` error "GFromActions.V1"

instance MonadFail b => GFromActions b V1 where
  gFromActions = fail "GFromActions.V1"

instance (Alternative t, GToActions t f, GToActions t g) => GToActions t (f :*: g) where
  gToActions (f :*: g) = gToActions @t f <|> gToActions @t g

instance (Monad b, GFromActions b f, GFromActions b g) => GFromActions b (f :*: g) where
  gFromActions = (:*:) <$> gFromActions @b <*> gFromActions @b

instance (Applicative t, Alternative t, GToActions t f, GToActions t g) => GToActions t (f :+: g) where
  gToActions (L1 f) = (pure L) <|> gToActions @t f
  gToActions (R1 g) = (pure R) <|> gToActions @t g

instance (MonadPlus b, GFromActions b f, GFromActions b g) => GFromActions b (f :+: g) where
  gFromActions = (is L >> L1 <$> gFromActions @b) <|> (is R >> R1 <$> gFromActions @b)

instance (Alternative t, ToActions t a, ToActions t b) => ToActions t (a, b)
instance (Monad b, FromActions b a, FromActions b b') => FromActions b (a, b')

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c) => ToActions t (a, b, c)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c) => FromActions b (a, b', c)

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c, ToActions t d) => ToActions t (a, b, c, d)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c, FromActions b d) => FromActions b (a, b', c, d)

instance (Alternative t, ToActions t a, ToActions t b, ToActions t c, ToActions t d, ToActions t e) => ToActions t (a, b, c, d, e)
instance (Monad b, FromActions b a, FromActions b b',  FromActions b c, FromActions b d, FromActions b e) => FromActions b (a, b', c, d, e)

instance (Applicative t, Alternative t, ToActions t a) => ToActions t [a] where
  toActions as = pure Grow <|> go as
    where go [] = pure Reduce
          go (a : as) = toActions a <|> go as
instance (MonadPlus b, FromActions b a) => FromActions b [a] where
  fromActions = is Grow >> manyTill (fromActions @b) (is Reduce)

instance (Applicative t, Alternative t, ToActions t a) => ToActions t (Maybe a) where
  toActions ma = pure Grow <|> go ma <|> pure Reduce
    where go Nothing = empty
          go (Just a) = toActions a
instance (MonadPlus b, FromActions b a) => FromActions b (Maybe a) where
  fromActions = is Grow >> option Nothing (Just <$> fromActions) >>= (is Reduce >>) . pure

instance (Applicative t, Alternative t, ToActions t a, ToActions t b) => ToActions t (Either a b)
instance (MonadPlus b, FromActions b a, FromActions b b') => FromActions b (Either a b')

instance Applicative t => ToActions t Text where
  toActions = pure . SToken

instance MonadFail b => FromActions b Text where
  fromActions = token >>= (\case SToken s -> pure s; _ -> fail "text")

instance Applicative t => ToActions t Int where
  toActions = pure . IToken

instance MonadFail b => FromActions b Int where
  fromActions = token >>= (\case IToken i -> pure i; _ -> fail "int")

instance Applicative t => ToActions t Bool where
  toActions = pure . BToken

instance MonadFail b => FromActions b Bool where
  fromActions = token >>= (\case BToken b -> pure b; _ -> fail "bool")

instance Alternative t => ToActions t () where
  toActions = const empty

instance Monad b => FromActions b () where
  fromActions = pure ()

-- | Runs the parser on the supplied input and returns whether or not the parse succeeded.
-- Results are discarded.
-- TODO: this isn't nice yet. It would be great if there was a stronger signal for failure than just 'mzero'.
-- Parser b i a ~ FreeT ((->) i) b a
checkParser :: forall b i a . MonadPlus b => Parser b i a -> i -> b ()
checkParser p i = do
  val <- runFreeT p
  case val of
    Pure a -> mzero
    Free f -> void . runFreeT $ f i

-- FreeT ((->) i) b a ~ StateT (b i) b a ???

-- iterTM specialized to Parser b i a ~ FreeT ((->) i) b a
-- iterTM :: (Monad b, MonadTrans t, Monad (t b)) => ((i -> t b a) -> t b a) -> Parser b i a -> t b a
-- iterTM f p = do
--   val <- lift . runFreeT $ p
--   case val of
--     Pure x -> return x
--     Free y -> f $ \i -> iterTM f (y i)

-- this version of @'iterTM'@ exposes the invermediate step
iterTM' :: (MonadTrans t, Monad b, Monad (t b)) => ((i -> Parser b i a) -> (Parser b i a -> t b a) -> t b a) -> Parser b i a -> t b a
iterTM' f p = do
  val <- lift . runFreeT $ p
  case val of
    Pure x -> return x
    Free y -> f y (iterTM' f)

-- | Idea here:
-- * instead of reducing @s@, we grow it, starting, e.g., from @[]@
-- * @i -> 'Parser' b i a@ is evaluated against the vocabulary, e.g. @[i]@, and only those @i@'s for which the parser does not fail are considered for continuation. among those, the model decides which to pick
-- * @s@ is the sequence of actions, @[i]@, and, at each step, we feed all previous actions to the model to get the next one
-- * in order to support the prediction, information about the parsing step is encoded in @b@
-- I shall test this idea with a random model.
-- How do to beam search?
-- does this work for training? I guess @next@ would build up a loss term. How do handle batching?
-- fresh values when backtracking: https://hackage.haskell.org/package/monad-gen-0.1.0.0/docs/Control-Monad-Gen.html
parse
  :: forall s b i a
   . Monad b
  => ((i -> Parser b i a) -> s -> b (Parser b i a, s))
  -> Parser b i a
  -> s
  -> b (a, s)
parse next =
  -- let f ip ps = StateT $ \s -> do
  --                 ~(p, s') <- next ip s
  --                 runStateT (ps p) s'
  let f ip ps = StateT (next ip) >>= ps
  in runStateT . iterTM' f

pures :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g a
pures = foldr (\x xs -> case x of Pure a -> pure a <|> xs; _ -> xs) empty

frees :: (Foldable g, Alternative g) => g (FreeF f a (FreeT f m a)) -> g (f (FreeT f m a))
frees = foldr (\x xs -> case x of Free fb -> pure fb <|> xs; _ -> xs) empty

batchedIterTM
  :: forall f t b a i
   . (Traversable f, Foldable f, Alternative f, MonadTrans t, Monad b, Monad (t b))
  => (f a -> f (i -> Parser b i a) -> (f (Parser b i a) -> t b (f a)) -> t b (f a))
  -> f (Parser b i a)
  -> t b (f a)
batchedIterTM f ps = do 
  vals <- traverse (lift @t . runFreeT) ps
  f (pures vals) (frees vals) (batchedIterTM f)

batchedParse
  :: forall f s b i a
   . (Traversable f, Foldable f, Alternative f, Monad b)
  => (f (i -> Parser b i a) -> s -> b (f (Parser b i a), s))
  -> f (Parser b i a)
  -> s
  -> b (f a, s)
batchedParse next = do
  let f as ip ps = StateT (next ip) >>= ps >>= (pure . (<|> as))
  runStateT . batchedIterTM f


data Stuff = SomeStuff { anInt :: Int, aBool :: Bool, moreStuff :: [Stuff], maybeFoo :: Maybe Foo }
          --  | NoStuff
  deriving (Eq, Show, Generic)

instance ToActions [] Stuff
instance FromActions [] Stuff

data Foo = SomeFoo { someText :: Text, stuff :: Stuff }
        --  | NoFoo
  deriving (Eq, Show, Generic)

instance ToActions [] Foo
instance FromActions [] Foo

test :: ([Action], [((Foo, [Action]), Env)], [((), Env)])
test =
  let env = defaultEnv
      stuff 0 = []
      stuff n = SomeStuff n True [] Nothing : stuff (n - 1)
      foo 0 = SomeFoo "a" $ SomeStuff 0 False [SomeStuff 2 True [] Nothing] Nothing
      foo n = SomeFoo "a" $ SomeStuff n ((==0) . (`rem` 3) $ n) [SomeStuff 2 False (stuff n) Nothing] (Just $ foo (n - 1))
      challenge = foo 2
      actions = toActions @[] challenge
      parser = fromActions @[]
      result' = let f ap [] = empty
                    f ap (a : as) = let p = ap a in do
                      env' <- get
                      pure $ unsafePerformIO $ print
                        ( a
                        , view (field @"meta") env'
                        , view (field @"pos") env'
                        , view (field @"rEnv" . field @"parentPos") env'
                        -- , view (field @"rEnv" . field @"relations") env'
                        ) >> pure (p, as)
                in runStateT (parse f parser actions) env
  in (actions, result', runStateT (checkParser parser (IToken 1)) env)

------------------------------------------------------------------------
-- A simply-typed lambda calculus with ints, bools, and strings
-- from https://github.com/hedgehogqa/haskell-hedgehog/blob/master/hedgehog-example/src/Test/Example/STLC.hs

-- input: randomly generated lambda calculus terms
-- output: reduced lambda calculus terms
-- accuracy based on exact match
-- variables need to be anonymized, use relations to indicate identical variables
-- constrain the production using `guard` on typecheck
-- how can I typecheck during parsing?
-- backtracking of FreeT ((->) i) [] a will solve it automatically

data TType =
    TBool
  | TInt
  | TString
  | TArrow TType TType
    deriving (Eq, Ord, Show, Generic)

instance ToActions [] TType
instance FromActions [] TType

-- | Lambda terms.
-- TODO: use De Bruijn indices https://en.wikipedia.org/wiki/De_Bruijn_index
-- https://stackoverflow.com/questions/28931477/haskell-convert-de-bruijn-terms-to-lambda-terms-and-vice-versa
-- https://hackage.haskell.org/package/lambda-sampler-1.1/docs/Data-Lambda.html
-- https://mroman42.github.io/mikrokosmos/haddock/Stlc-Types.html
-- https://www.schoolofhaskell.com/user/edwardk/bound
-- https://github.com/ekmett/bound/blob/master/examples/Simple.hs
-- http://hackage.haskell.org/package/bound-gen
-- http://hackage.haskell.org/package/bound-extras
-- https://github.com/robrix/path/blob/master/src/Path/Term.hs
-- https://en.wikipedia.org/wiki/Programming_Computable_Functions
-- https://github.com/jozefg/pcf/blob/master/src/Language/Pcf.hs
-- https://jozefg.bitbucket.io/posts/2014-12-17-variables.html
-- http://blog.ielliott.io/topsy-turvy-reverse-state/
data Expr =
    EBool Bool
  | EInt Int
  | EString Text
  | EVar Text
  | ELam Text TType Expr
  | EApp Expr Expr
    deriving (Eq, Ord, Show, Generic)

instance ToActions [] Expr
instance FromActions [] Expr

------------------------------------------------------------------------

-- | de Bruijn indices in unary notation.
-- data BIndex =
--     BZero
--   | BSucc BIndex
--     deriving (Eq, Ord, Show, Generic)

-- instance ToActions [] BIndex
-- instance FromActions [] BIndex

-- | Translates the given index to a corresponding positive integer.
-- toInt :: (Num a) => BIndex -> a
-- toInt BZero = 0
-- toInt (BSucc n) = 1 + toInt n

-- | Translates the given positive integer to a corresponding index.
-- toIndex :: (Num a, Eq a) => a -> BIndex
-- toIndex 0 = BZero
-- toIndex n = BSucc $ toIndex (n-1)

-- | Lambda terms with de Bruijn indices.
-- data BExpr =
--     BBool Bool
--   | BInt Int
--   | BString Text
--   | BVar BIndex -- ^ de Bruijn indices.
--   | BLam TType BExpr -- ^ Lambda abstraction.
--   | BApp BExpr BExpr -- ^ Term application.
--     deriving (Eq, Ord, Show, Generic)

-- instance ToActions [] BExpr
-- instance FromActions [] BExpr

------------------------------------------------------------------------

data Ty =
    Arr Ty Ty
  | Nat
  deriving (Eq, Ord, Show, Generic)

instance ToActions [] Ty
instance FromActions [] Ty

infixl 9 :@
data Exp a =
    Var a -- ^ Variable.
  | Lam Ty (Scope () Exp a) -- ^ Lambda abstraction.
  | Exp a :@ Exp a -- ^ Term application.
  | Suc (Exp a)
  | Zero
  deriving (Functor, Foldable, Traversable, Generic)

-- https://github.com/ekmett/bound/commit/6ab44d9a84a24e376bf29548ea61c1baf9ce3063
-- instance Eq1 Exp where
--   liftEq eq (Var a) (Var b) = eq a b
--   liftEq eq (Lam ty b) (Lam ty' b') = ty == ty' && liftEq eq b b'
--   liftEq eq (a :@ a') (b :@ b') = liftEq eq a b && liftEq eq a' b'
--   liftEq eq (Suc a) (Suc b) = liftEq eq a b
--   liftEq _ Zero Zero = True
--   liftEq _ _ _ = False


-- instance Ord1 Exp where
--   liftCompare comp (Var a) (Var b) = comp a b
--   liftCompare _ (Var _) (Lam _ _) = LT
--   liftCompare _ (Var _) (_ :@ _) = LT
--   liftCompare _ (Var _) (Suc _) = LT
--   liftCompare _ (Var _) Zero = LT
--   liftCompare _ (Lam _ _) (Var _) = GT
--   liftCompare comp (Lam ty b) (Lam ty' b') = compare ty ty' `mappend` liftCompare comp b b'
--   liftCompare _ (Lam _ _) (_ :@ _) = LT
--   liftCompare _ (Lam _ _) (Suc _) = LT
--   liftCompare _ (Lam _ _) Zero = LT
--   ...

-- instance Show1 Exp where
--   liftShowsPrec s _ d (Var a) = showParen (d > 10) $ showString "Var " . s 11 a
--   liftShowsPrec s sl d (a :@ b) = showParen (d > 9) $ liftShowsPrec s sl 9 a . showString " :@ " . liftShowsPrec s sl 10 b
--   liftShowsPrec s sl d (Lam _ b) = showParen (d > 10) $ showString "Lam " . liftShowsPrec s sl 11 b
--   liftShowsPrec s sl d (Suc b) = showParen (d > 10) $ showString "Suc " . liftShowsPrec s sl 11 b
--   liftShowsPrec _ _ d Zero = showParen (d > 10) $ showString "Zero"

instance Applicative Exp where
  pure = Var
  (<*>) = ap

instance Monad Exp where
  return = Var
  Var a >>= f = f a
  (x :@ y) >>= f = (x >>= f) :@ (y >>= f)
  Lam ty e >>= f = Lam ty (e >>>= f)
  Suc e >>= f = Suc (e >>= f)
  Zero >>= _ = Zero

deriveEq1 ''Exp
deriveOrd1 ''Exp
deriveShow1 ''Exp

instance Eq a => Eq (Exp a) where (==) = eq1
instance Ord a => Ord (Exp a) where compare = compare1
instance Show a => Show (Exp a) where showsPrec = showsPrec1

instance ToActions [] a => ToActions [] (Var () (Exp a))
instance ToActions [] a => ToActions [] (Scope () Exp a)
instance ToActions [] a => ToActions [] (Exp a)

instance FromActions [] a => FromActions [] (Var () (Exp a))
instance FromActions [] a => FromActions [] (Scope () Exp a)
instance FromActions [] a => FromActions [] (Exp a)

lam :: Eq a => Ty -> a -> Exp a -> Exp a
lam ty varName b = Lam ty (abstract1 varName b)

-- | Smart constructor that converts the given positive integer to a corresponding Nat.
nat :: (Num n, Eq n) => n -> Exp a
nat 0 = Zero
nat n = Suc $ nat (n-1)

-- | Compute the normal form of an expression.
nf :: Exp a -> Exp a
nf e@Var{} = e
nf (Lam ty b) = Lam ty (toScope . nf . fromScope $ b)
nf (f :@ a) = case whnf f of
  Lam ty b -> nf (instantiate1 a b)
  f' -> nf f' :@ nf a
nf e@Suc{} = e
nf e@Zero = e

-- | Reduce a term to weak head normal form.
whnf :: Exp a -> Exp a
whnf e@Var{} = e
whnf e@Lam{} = e
whnf (f :@ a) = case whnf f of
  Lam _ b -> whnf (instantiate1 a b)
  f' -> f' :@ a
whnf e@Suc{} = e
whnf e@Zero = e

type TyM a = MaybeT (Fresh a)

assertTy :: Ord a => Map a Ty -> Exp a -> Ty -> TyM a ()
assertTy env e t = (== t) <$> typeCheck env e >>= guard

typeCheck :: Ord a => Map a Ty -> Exp a -> TyM a Ty
typeCheck _ Zero = return Nat
typeCheck env (Suc e) = assertTy env e Nat >> return Nat
typeCheck env (Var a) = MaybeT . return $ Map.lookup a env
typeCheck env (f :@ a) = typeCheck env f >>= \case
  Arr fTy tTy -> assertTy env a fTy >> return tTy
  _ -> mzero
typeCheck env (Lam ty bind) = do
  varName <- fresh
  Arr ty <$> typeCheck (Map.insert varName ty env) (instantiate1 (Var varName) bind)

testBound :: IO ()
testBound = do
  let term :: Exp Int = (lam (Nat) 0 (Var 0)) :@ Zero
  print term -- Lam Nat (Scope (Var (B ()))) :@ Zero
  print $ runFresh . runMaybeT . typeCheck Map.empty $ term -- Just Nat
  print $ toActions @[] $ term -- [R,L,L,R,R,L,L,L,R,R,R]
  print $ nf term -- Zero
  print $ toActions @[] $ nf term -- [R,R,R]

instance MFunctor (FreshT e) where
  hoist nat m = FreshT $ hoist (hoist nat) (unFreshT m)

instance MonadGen m => MonadGen (FreshT e m) where
  type GenBase (FreshT e m) = FreshT e (GenBase m)
  toGenT = hoist FreshT . distributeT . hoist distributeT . unFreshT . hoist toGenT
  fromGenT = hoist fromGenT . distributeT

genTy :: MonadGen m => m Ty
genTy =
  Gen.recursive Gen.choice [
      pure Nat
    ] [
      Arr <$> genTy <*> genTy
    ]

genWellTypedExp :: (Eq a, Enum a) => Ty -> Gen (Exp a)
genWellTypedExp ty = runFreshT $ runReaderT (genWellTypedExp' ty) mempty

genWellTypedExp' :: Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp' ty =
  Gen.shrink shrinkExp $
  Gen.recursive Gen.choice [
      genWellTypedExp'' ty
    ] [
      genWellTypedPath' ty <|> genWellTypedApp' ty
    , genWellTypedApp' ty
    ]

shrinkExp :: Exp a -> [Exp a]
shrinkExp (f :@ a) = case whnf f of
  Lam _ b -> [whnf (instantiate1 a b)]
  _ -> []
shrinkExp _ = []

type GTyM a = ReaderT (Map Ty [Exp a]) (FreshT a Gen)

genWellTypedExp'' :: Eq a => Ty -> GTyM a (Exp a)
genWellTypedExp'' (Arr ty ty') = do
  varName <- fresh
  lam ty varName <$> local (insertVar' varName ty) (genWellTypedExp' ty')
genWellTypedExp'' Nat = nat <$> Gen.int (Range.linear 0 10)

insertVar' :: Eq a => a -> Ty -> Map Ty [Exp a] -> Map Ty [Exp a]
insertVar' varName ty =
  Map.insertWith (<>) ty [Var varName] . fmap (List.filter (/= Var varName))

genWellTypedApp' :: Eq a => Ty -> GTyM a (Exp a)
genWellTypedApp' ty = do
  tg <- genKnownTypeMaybe'
  eg <- genWellTypedExp' tg
  let tf = Arr tg ty
  ef <- genWellTypedExp' tf
  pure (ef :@ eg)

-- | This tries to look up a known expression of the desired type from the env.
-- It does not always succeed, throwing `empty` when unavailable.
genWellTypedPath' :: Ty -> GTyM a (Exp a)
genWellTypedPath' ty = do
  paths <- ask
  case fromMaybe [] (Map.lookup ty paths) of
    [] -> empty
    es -> Gen.element es

genKnownTypeMaybe' :: GTyM a Ty
genKnownTypeMaybe' = do
  known <- ask
  if Map.null known then
    genTy
  else
    Gen.frequency [
        (2, Gen.element $ Map.keys known)
      , (1, genTy)
      ]

------------------------------------------------------------------------

data Example a b = Example
  { input :: Input a
  , target :: Target b
  } deriving (Eq, Ord, Show, Generic)

instance (Alternative t, ToActions t a, ToActions t b) => ToActions t (Example a b)
instance (Monad b, FromActions b a, FromActions b b') => FromActions b (Example a b')

newtype Input  a = Input  a deriving (Eq, Ord, Show, Generic)
newtype Target a = Target a deriving (Eq, Ord, Show, Generic)

instance ToActions t a => ToActions t (Input a)
instance (Monad b, FromActions b a) => FromActions b (Input a) where
  fromActions = do
    modify $ field @"aEnv" . field @"currentScope" .~ pure "input"
    modify $ field @"aEnv" . field @"knownScopes" %~ go "input"
    Input <$> fromActions
    where go scopeId = Map.insert scopeId (AttentionScope BidirectionalAttention (Set.singleton "input") mempty)

instance ToActions t a => ToActions t (Target a)
instance (Monad b, FromActions b a) => FromActions b (Target a) where
  fromActions = do
    modify $ field @"aEnv" . field @"currentScope" .~ pure "target"
    modify $ field @"aEnv" . field @"knownScopes" %~ go "target"
    Target <$> fromActions
    where go scopeId = Map.insert scopeId (AttentionScope BackwardAttention (Set.fromList ["input", "target"]) mempty)

------------------------------------------------------------------------

-- -- | Evaluate to weak head normal form.
-- evaluate :: Expr -> Expr
-- evaluate expr =
--   case expr of
--     EBool _ ->
--       expr
--     EInt _ ->
--       expr
--     EString _ ->
--       expr
--     EVar _ ->
--       expr
--     ELam _ _ _ ->
--       expr
--     EApp f g ->
--       case evaluate f of
--         ELam x _t e ->
--           evaluate (subst x g e)
--         h ->
--           EApp h g

-- subst :: Text -> Expr -> Expr -> Expr
-- subst x y expr =
--   case expr of
--     EBool _ ->
--       expr
--     EInt _ ->
--       expr
--     EString _ ->
--       expr
--     EVar z ->
--       if x == z then
--         y
--       else
--         expr
--     ELam n t g ->
--       if n == x then
--         ELam n t g
--       else
--         ELam n t (subst x y g)
--     EApp f g ->
--       EApp (subst x y f) (subst x y g)

-- -- | Collect all the free variables in an 'Expr'.
-- free :: Expr -> Set Text
-- free =
--   free' mempty mempty

-- free' :: Set Text -> Set Text -> Expr -> Set Text
-- free' binds frees expr =
--   case expr of
--     EBool _ ->
--       frees
--     EInt _ ->
--       frees
--     EString _ ->
--       frees
--     EVar x ->
--       if Set.member x binds then
--         frees
--       else
--         Set.insert x frees
--     ELam x _t y ->
--       free' (Set.insert x binds) frees y
--     EApp f g ->
--       free' binds frees f <> free' binds frees g

-- ------------------------------------------------------------------------

-- data TypeError =
--     Mismatch TType TType
--   | FreeVariable Text
--   | ExpectedArrow TType
--   deriving (Eq, Ord, Show)

-- -- | Typecheck some expression.
-- typecheck :: Expr -> Either TypeError TType
-- typecheck =
--   typecheck' mempty

-- typecheck' :: Map Text TType -> Expr -> Either TypeError TType
-- typecheck' env expr =
--   case expr of
--     EBool _ ->
--       pure TBool

--     EInt _ ->
--       pure TInt

--     EString _ ->
--       pure TString

--     EVar x ->
--       maybe (Left (FreeVariable x)) pure (Map.lookup x env)

--     ELam x t y ->
--       TArrow t <$> typecheck' (Map.insert x t env) y

--     EApp f g -> do
--       tf <- typecheck' env f
--       tg <- typecheck' env g
--       case tf of
--         TArrow ta tb ->
--           if ta == tg then
--             pure tb
--           else
--             Left (Mismatch ta tg)
--         _ ->
--           Left (ExpectedArrow tf)

-- ------------------------------------------------------------------------

-- genType :: MonadGen m => m TType
-- genType =
--   Gen.recursive Gen.choice [
--       pure TBool
--     , pure TInt
--     , pure TString
--     ] [
--       TArrow <$> genType <*> genType
--     ]

-- ------------------------------------------------------------------------

-- genWellTypedExpr :: TType -> Gen Expr
-- genWellTypedExpr =
--   flip runReaderT mempty . genWellTypedExpr'

-- genWellTypedExpr' :: TType -> ReaderT (Map TType [Expr]) Gen Expr
-- genWellTypedExpr' want =
--   Gen.shrink shrinkExpr $
--   Gen.recursive Gen.choice [
--       genWellTypedExpr'' want
--     ] [
--       genWellTypedPath want <|> genWellTypedApp want
--     , genWellTypedApp want
--     ]

-- shrinkExpr :: Expr -> [Expr]
-- shrinkExpr expr =
--   case expr of
--     EApp f g ->
--       case evaluate f of
--         ELam x _ e ->
--           [evaluate (subst x g e)]
--         _ ->
--           []
--     _ ->
--       []

-- genWellTypedExpr'' :: TType -> ReaderT (Map TType [Expr]) Gen Expr
-- genWellTypedExpr'' want =
--   case want of
--     TBool ->
--       EBool <$> Gen.element [True, False]
--     TInt ->
--       EInt <$> Gen.int (Range.linear 0 10000)
--     TString ->
--       EString <$> Gen.text (Range.linear 0 25) Gen.lower
--     TArrow t1 t2 -> do
--       x <- Gen.text (Range.linear 1 25) Gen.lower
--       ELam x t1 <$> local (insertVar x t1) (genWellTypedExpr' t2)

-- insertVar :: Text -> TType -> Map TType [Expr] -> Map TType [Expr]
-- insertVar n typ =
--   Map.insertWith (<>) typ [EVar n] .
--   fmap (List.filter (/= EVar n))

-- genWellTypedApp :: TType -> ReaderT (Map TType [Expr]) Gen Expr
-- genWellTypedApp want = do
--   tg <- genKnownTypeMaybe
--   eg <- genWellTypedExpr' tg
--   let tf = TArrow tg want
--   ef <- genWellTypedExpr' tf
--   pure (EApp ef eg)

-- -- | This tries to look up a known expression of the desired type from the env.
-- -- It does not always succeed, throwing `empty` when unavailable.
-- genWellTypedPath :: TType -> ReaderT (Map TType [Expr]) Gen Expr
-- genWellTypedPath want = do
--   paths <- ask
--   case fromMaybe [] (Map.lookup want paths) of
--     [] ->
--       empty
--     es ->
--       Gen.element es

-- genKnownTypeMaybe :: ReaderT (Map TType [Expr]) Gen TType
-- genKnownTypeMaybe = do
--   known <- ask
--   if Map.null known then
--     genType
--   else
--     Gen.frequency [
--         (2, Gen.element $ Map.keys known)
--       , (1, genType)
--       ]

-- ------------------------------------------------------------------------

-- -- Generates a term that is ill-typed at some point.
-- genIllTypedExpr :: Gen Expr
-- genIllTypedExpr = do
--   be <- genIllTypedApp
--   Gen.recursive Gen.choice [
--       -- Don't grow - just dish up the broken expr
--       pure be
--     ] [
--       -- Grow a reasonable app expression around the error
--       do tg <- genType
--          tf <- genType
--          let ta = TArrow tg tf
--          ea <- genWellTypedExpr ta
--          pure (EApp ea be)
--     ]

-- -- Generates a term that is ill-typed at the very top.
-- genIllTypedApp :: Gen Expr
-- genIllTypedApp = do
--   t1 <- genType
--   t2 <- genType
--   t3 <- genType
--   guard (t1 /= t2)
--   f <- genWellTypedExpr t3
--   g <- genWellTypedExpr t2
--   x <- Gen.text (Range.linear 1 25) Gen.lower
--   pure $ EApp (ELam x t1 f) g

------------------------------------------------------------------------

prep :: PropertyT IO (Ty, Example (Exp Int) (Exp Int), [((Example (Exp Int) (Exp Int), [Action]), Env)])
prep = do
  ty <- forAll genTy
  input <- forAll (genWellTypedExp ty)
  let target = nf input
      ex = Example (Input input) (Target target)
      env = defaultEnv
      actions = toActions @[] ex
  guard (length actions <= 512)
  let parser = fromActions @[] @(Example (Exp Int) (Exp Int))
      result = let f ap [] = empty
                   f ap (a : as) = let p = ap a in pure (p, as)
               in  runStateT (parse f parser actions) env
  pure (ty, ex, result)

prop_welltyped :: Property
prop_welltyped =
  property $ do
    (ty, Example (Input input) (Target target), _) <- prep
    let (Just ty') = runFresh . runMaybeT . typeCheck Map.empty $ input
    let (Just ty'') = runFresh . runMaybeT . typeCheck Map.empty $ target
    ty === ty'
    ty === ty''

-- test that every position belongs only to at most one attention scope
prop_attentionScope :: Property
prop_attentionScope = property $ do
  (_, _, [(_, Env {..})]) <- prep
  let r = Map.elems $ scopePositions <$> knownScopes aEnv
      c = sort . join $ Set.toList <$> r
      u = Set.toList . Set.unions $ r
  c === u

-- test presence of self attention
prop_selfAttention :: Property
prop_selfAttention = property $ do
  (_, _, [(_, Env {..})]) <- prep
  let sa = foldr (\(pos, pos') -> \b -> if pos == pos' then Set.insert pos b else b) Set.empty (attentionMask aEnv)
  sa === keyPaddingMask aEnv

-- test round trip serialization-deserialization
prop_roundTrip :: Property
prop_roundTrip = property $ do
  (_, ex, [((reconstructedEx, _), _)]) <- prep
  ex === reconstructedEx

testSTLC :: IO Bool
testSTLC = checkParallel $$(discover)

-------------------------

newtype Successor a = Successor {suc :: a -> a}

-- | The monad transformer for generating fresh values.
data FreshT e m a = FreshT { unFreshT :: ReaderT (Successor e) (StateT e m) a }
                    deriving (Functor)

instance Monad m => MonadFresh e (FreshT e m) where
  fresh = FreshT $ do
    s <- asks suc
    modify s
    get

instance Monad m => Monad (FreshT e m) where
  return = FreshT . return
  (FreshT m) >>= f = FreshT $ m >>= unFreshT . f
instance MonadPlus m => MonadPlus (FreshT e m) where
  mzero = FreshT mzero
  mplus (FreshT m) (FreshT m') = FreshT $ mplus m m'
instance (Functor f, Monad f) => Applicative (FreshT e f) where
  pure = FreshT . pure
  (FreshT f) <*> (FreshT a) = FreshT $ f <*> a
instance (Monad m, Functor m, MonadPlus m) => Alternative (FreshT e m) where
  empty = mzero
  (<|>) = mplus

type Fresh e = FreshT e Identity

instance MonadTrans (FreshT e) where
  lift = FreshT . lift . lift

instance MonadReader r m => MonadReader r (FreshT e m) where
  local f m = FreshT $ ask >>= lift . local f . runReaderT (unFreshT m)
  ask     = FreshT (lift ask)
instance MonadState s m => MonadState s (FreshT e m) where
  get    = FreshT $ (lift . lift) get
  put    = FreshT . lift . lift . put
instance (MonadWriter w m) => MonadWriter w (FreshT e m) where
  tell m = lift $ tell m
  listen = FreshT . listen . unFreshT
  pass   = FreshT . pass . unFreshT
instance MonadFix m => MonadFix (FreshT e m) where
  mfix = FreshT . mfix . (unFreshT .)
instance MonadIO m => MonadIO (FreshT e m) where
  liftIO = FreshT . liftIO
instance MonadCont m => MonadCont (FreshT e m) where
  callCC f = FreshT $ callCC (unFreshT . f . (FreshT .))
instance MonadError e m => MonadError e (FreshT e' m) where
  throwError = FreshT . throwError
  catchError m h = FreshT $ catchError (unFreshT m) (unFreshT . h)

successor :: (e -> e) -> Successor e
successor = Successor

enumSucc :: Enum e => Successor e
enumSucc = Successor succ

-- | Run a @GenT@ computation starting from the value
-- @toEnum 0@
runFreshT :: (Enum e, Monad m) => FreshT e m a -> m a
runFreshT = runFreshTFrom (toEnum 0)

-- | Run a @Gen@ computation starting from the value
-- @toEnum 0@
runFresh :: Enum e => Fresh e a -> a
runFresh = runFreshFrom (toEnum 0)

-- | Run a @FreshT@ computation starting from a specific value @e@.
runFreshTFrom :: (Monad m, Enum e) => e -> FreshT e m a -> m a
runFreshTFrom e = runFreshTWith enumSucc e

-- | Run a @Fresh@ computation starting from a specific value @e@.
runFreshFrom :: Enum e => e -> Fresh e a -> a
runFreshFrom e = runFreshWith enumSucc e

-- | Run a @FreshT@ computation starting from a specific value @e@ with
-- a the next fresh value determined by @Successor e@.
runFreshTWith :: Monad m => Successor e -> e -> FreshT e m a -> m a
runFreshTWith s e = flip evalStateT e
                  . flip runReaderT s
                  . unFreshT

-- | Run a @FreshT@ computation starting from a specific value @e@ with
-- a the next fresh value determined by @Successor e@.
runFreshWith :: Successor e -> e -> Fresh e a -> a
runFreshWith s e = runIdentity . runFreshTWith s e

---------------------------

-- | The MTL style class for generating fresh values
class Monad m => MonadFresh e m | m -> e where
  -- | Generate a fresh value @e@, @fresh@ should never produce the
  -- same value within a monadic computation.
  fresh :: m e

instance MonadFresh e m => MonadFresh e (IdentityT m) where
  fresh = lift fresh
instance MonadFresh e m => MonadFresh e (StateT s m) where
  fresh = lift fresh
instance MonadFresh e m => MonadFresh e (ReaderT s m)  where
  fresh = lift fresh
instance (MonadFresh e m, Monoid s) => MonadFresh e (WriterT s m)  where
  fresh = lift fresh
instance MonadFresh e m => MonadFresh e (MaybeT m) where
  fresh = lift fresh
instance MonadFresh e m => MonadFresh e (ContT r m) where
  fresh = lift fresh
instance (Monoid w, MonadFresh e m) => MonadFresh e (RWST r w s m) where
  fresh = lift fresh
instance MonadFresh e m => MonadFresh e (SS.StateT s m) where
  fresh = lift fresh
instance (Monoid w, MonadFresh e m) => MonadFresh e (SW.WriterT w m) where
  fresh = lift fresh
instance (MonadFresh e m) => MonadFresh e (ExceptT e' m) where
  fresh = lift fresh
