{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Module defining typeclasses for creating models, training
-- and monitoring routines using (concurrent) streamly streams.
-- Instances for these typeclasses should be implemented along with your model,
-- not here.
module Trainer where

import Control.Lens.TH (makePrisms)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Control ()
import Data.Functor.Const (Const (Const))
import Data.Functor.Identity (Identity)
import Data.Functor.Product (Product)
import Data.Kind (Type)
import qualified Data.Monoid.Statistics as StatM
  ( MeanKBN,
    StatMonoid,
    Variance,
    addValue,
    singletonMonoid,
  )
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import Graphics.Vega.VegaLite (VegaLite, toHtmlFile)
import Streamly (IsStream, SerialT)
import qualified Streamly.Data.Fold as FL
import Streamly.Internal.Data.Fold (mkFold)
import Streamly.Prelude as S
  ( concatMap,
    concatMapM,
    fold,
    fromList,
    map,
    mapM,
    postscanlM',
    yield,
  )
import qualified Torch.DType as D
import qualified Torch.Internal.Class as ATen
import qualified Torch.Tensor as D
import Torch.Typed
  ( Apply,
    BasicArithmeticDTypeIsValid,
    ComparisonDTypeIsValid,
    HFoldrM,
    HList,
    HMap',
    HMapM',
    HUnfoldM,
    HUnfoldMRes,
    HasGrad,
    KnownDevice,
    MakeIndependent,
    MonadBaseControl,
    Optimizer,
    Parameterized,
    RandDTypeIsValid,
    StandardFloatingPointDTypeValidation,
    SumDTypeIsValid,
    Tensor (toDynamic),
    TensorListFold,
    TensorListUnfold,
    ToDependent,
    ZerosLike,
    toCPU,
  )

type Two f a = Product f f a

class Monitorable metric view where
  render :: metric -> view

asFloat :: forall device. Tensor device 'D.Float '[] -> Float
asFloat = D.asValue . toDynamic . toCPU

newtype YHat model = YHat {unYHat :: Codomain model}

-- | This is a functor from finite vector space to category of metric ... spaces?
-- Would be nice to actually do the proof . . .
class HasMetricSpace someModel where
  type Domain someModel :: Type
  type Codomain someModel :: Type
  type MetricSpace (f :: Type -> Type) someModel :: Type

  measure ::
    proxy someModel ->
    YHat someModel ->
    Codomain someModel ->
    MetricSpace Identity someModel

  propagationStepM ::
    ( MonadIO m
    ) =>
    someModel ->
    Bool ->
    Domain someModel ->
    m (Codomain someModel)

  inferenceStepM ::
    ( MonadIO m
    ) =>
    someModel ->
    Domain someModel ->
    m (YHat someModel)
  inferenceStepM m = fmap YHat . propagationStepM m False

  stochasticStepM ::
    ( MonadIO m
    ) =>
    someModel ->
    Domain someModel ->
    m (YHat someModel)
  stochasticStepM m = fmap YHat . propagationStepM m True

  validationStream :: (IsStream t, MonadThrow m, MonadBaseControl IO m, MonadIO m, Monad (t m)) => Bool -> someModel -> t m (Domain someModel, Codomain someModel) -> t m (MetricSpace Identity someModel)
  validationStream isStoch m = S.mapM (go m)
    where
      stepFn m' x'
        | isStoch = stochasticStepM m' x'
        | otherwise = inferenceStepM m' x'
      go m' (x, y) = do
        yhat <- liftIO $ stepFn m' x
        pure $ measure (Proxy :: Proxy someModel) yhat y

  {-# MINIMAL measure, propagationStepM #-}

class HasTrainState someModel optim where
  data TrainState someModel optim
  type TrainReport (f :: Type -> Type) model optim :: Type
  getModel :: TrainState someModel optim -> someModel
  getOptim :: TrainState someModel optim -> optim
  getTrainReport :: TrainState someModel optim -> TrainReport Identity someModel optim
  initTrainState :: someModel -> optim -> TrainState someModel optim

  {-# MINIMAL getModel, getOptim, getTrainReport, initTrainState #-}

class
  ( forall param tensors device gradients.
    Optimizable model param tensors device optim gradients =>
    (Parameterized model),
    HasTrainState model optim,
    HasMetricSpace model
  ) =>
  Trainable model optim
  where
  trainStepM ::
    ( MonadIO m
    ) =>
    TrainState model optim ->
    (Domain model, Codomain model) ->
    m (TrainState model optim)

  trainStream :: (IsStream t, MonadIO m, Monad (t m)) => model -> optim -> t m (Domain model, Codomain model) -> t m (TrainReport Identity model optim)
  trainStream m o s = tuckMonad . return . S.map getTrainReport $ S.postscanlM' trainStepM (initTrainState m o) s

  {-# MINIMAL trainStepM #-}

{-# INLINE tuckMonad #-}
tuckMonad :: (IsStream t, Monad m, Monad (t m)) => m (t m b) -> t m b
tuckMonad x = S.concatMapM (const x) (return ())

type SufficientlyCapable device =
  ( KnownDevice device,
    RandDTypeIsValid device 'D.Float,
    SumDTypeIsValid device 'D.Bool,
    StandardFloatingPointDTypeValidation device 'D.Float,
    ComparisonDTypeIsValid device 'D.Int64,
    BasicArithmeticDTypeIsValid device 'D.Float
  )

type Optimizable model params tensors device optim gradients =
  ( HasGrad (HList params) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent params tensors,
    ATen.Castable (HList gradients) [D.ATenTensor],
    Parameterized model,
    Optimizer optim gradients tensors 'D.Float device,
    HMap' ZerosLike params gradients,
    HMapM' IO MakeIndependent tensors params,
    HFoldrM IO TensorListFold [D.ATenTensor] gradients [D.ATenTensor],
    Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] gradients),
    HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] gradients) gradients
  )

type IsTrainable model params tensors device opt gradients =
  ( SufficientlyCapable device,
    HasTrainState model opt,
    KnownDevice device,
    Optimizable model params tensors device opt gradients
  )

data Report a b = Validation a | Batch b
  deriving stock (Eq, Ord, Show, Generic)

trainer ::
  forall f g model t optim m.
  ( Trainable model optim,
    MonadIO m,
    IsStream t,
    Monad (t m),
    Monoid (MetricSpace f model),
    MonadThrow m,
    MonadBaseControl IO m
  ) =>
  (MetricSpace Identity model -> MetricSpace f model) ->
  (TrainReport Identity model optim -> TrainReport g model optim) ->
  model ->
  optim ->
  t m (Domain model, Codomain model) ->
  SerialT m (Domain model, Codomain model) ->
  t m (Report (MetricSpace f model) (TrainReport g model optim))
trainer validationAggf trainAggf m o ts vs = S.concatMap (trainAndValidReport m o) dataCycle
  where
    dataCycle = S.fromList [Just ts, Nothing] <> dataCycle
    trainAndValidReport m' o' (Just s') = S.map (Batch . trainAggf) $ trainStream m' o' s'
    trainAndValidReport m' _ Nothing =
      S.map Validation . tuckMonad $
        fmap S.yield (S.fold FL.mconcat (S.map validationAggf $ validationStream False m' vs))

vegaLiteFold :: (Monitorable [Report a b] VegaLite, MonadIO m) => FilePath -> FL.Fold m (Report a b) ()
vegaLiteFold fp = mkFold go (return mempty) outM
  where
    outM as = liftIO $ toHtmlFile fp $ render as
    go as r = pure (r : as)

vegaLiteMonitor ::
  ( MonadIO m,
    Monitorable [Report a b] VegaLite
  ) =>
  FilePath ->
  SerialT m (Report a b) ->
  m ()
vegaLiteMonitor fp = S.fold (vegaLiteFold fp)

-- | Wrappers to get declarative monoidal stats that jive with hkds
newtype Mean a = Mean {unMean :: StatM.MeanKBN}
  deriving stock (Eq, Generic)
  deriving newtype (Monoid, Semigroup)

instance StatM.StatMonoid (Mean (Tensor device 'D.Float '[])) Float where
  addValue (Mean smp) !x = Mean (StatM.addValue smp x)
  singletonMonoid x = Mean (StatM.singletonMonoid x)
  {-# INLINE addValue #-}
  {-# INLINE singletonMonoid #-}

instance (Real x) => StatM.StatMonoid (Mean a) x where
  addValue (Mean smp) !x = Mean (StatM.addValue smp x)
  singletonMonoid x = Mean (StatM.singletonMonoid x)
  {-# INLINE addValue #-}
  {-# INLINE singletonMonoid #-}

newtype Variance a = Variance {unVariance :: StatM.Variance}
  deriving stock (Eq, Generic)
  deriving newtype (Monoid, Semigroup)

instance (Real x) => StatM.StatMonoid (Variance a) x where
  addValue (Variance smp) !x = Variance (StatM.addValue smp x)
  singletonMonoid x = Variance (StatM.singletonMonoid x)
  {-# INLINE addValue #-}
  {-# INLINE singletonMonoid #-}

instance StatM.StatMonoid m a => StatM.StatMonoid (Const m x) a where
  addValue (Const smp) !x = Const (StatM.addValue smp x)
  singletonMonoid x = Const (StatM.singletonMonoid x)
  {-# INLINE addValue #-}
  {-# INLINE singletonMonoid #-}

makePrisms ''Report