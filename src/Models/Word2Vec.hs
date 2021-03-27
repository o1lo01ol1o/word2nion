{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Models.Word2Vec where

import Barbies
  ( AllBF,
    ApplicativeB,
    Barbie (Barbie),
    ConstraintsB,
    FunctorB,
    Rec (Rec),
    TraversableB,
  )
import Control.Exception.Safe (MonadThrow)
import Control.Lens ((^.), (^?), _1, _2)
import Control.Monad ((<=<))
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Identity (Identity (runIdentity))
import Control.Monad.Trans.Control (MonadBaseControl)
import Data.Data (Proxy (Proxy))
import Data.Functor.Identity (Identity (Identity))
import Data.Functor.Product (Product (..))
import Data.List (foldl')
import Data.Maybe (mapMaybe)
import qualified Data.Monoid.Statistics as StatM
import Data.Sequence (Seq)
import qualified Data.Sequence as Seq
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Mod, Nat, type (-))
import GHC.TypeNats (type (<=?))
import Graphics.Vega.VegaLite
  ( DataValue (Number),
    Mark (Bar, Line),
    Measurement (Quantitative, Temporal),
    Operation (Count),
    Position (X, Y),
    PositionChannel (PAggregate, PBin, PName, PmType),
    VegaLite,
    dataFromRows,
    dataRow,
    encoding,
    height,
    mark,
    position,
    toVegaLite,
    width,
  )
import Streamly (IsStream, SerialT)
import qualified Torch.DType as D
import Torch.Functional as F (Dim (Dim), cat, cosineSimilarity')
import Torch.HList (HList, HMap', HMapM', HZipWith3)
import Torch.Initializers
  ( FanMode (FanIn),
    NonLinearity (LeakyRelu),
    kaimingUniform,
  )
import Torch.Typed (Adam, AdamBiasAdjustment, AdamMomentum1Update, AdamMomentum2Update, AdamParameterUpdate, All, AllDimsPositive, Embedding, EmbeddingSpec (LearnedEmbeddingWithCustomInitSpec), EmbeddingType (Learned), HZipWith, HasForward (forward), KnownDType, KnownDevice, MeanDTypeValidation, Parameter, RandDTypeIsValid, StandardFloatingPointDTypeValidation, SumDType, SumDTypeIsValid, Tensor (..), ZerosLike, meanAll, meanDim, natValI, powScalar, reshape, runStep, toFloat, type (>=))
import Torch.Typed.Autograd (HasGrad)
import Torch.Typed.Parameter
  ( MakeIndependent,
    Parameterized (Parameters),
    Randomizable (..),
    ToDependent,
  )
import Trainer
  ( HasMetricSpace
      ( Codomain,
        Domain,
        MetricSpace,
        measure,
        propagationStepM,
        stochasticStepM
      ),
    HasTrainState (..),
    IsTrainable,
    Mean,
    Monitorable (..),
    Optimizable,
    Report,
    SufficientlyCapable,
    Trainable (trainStepM),
    Variance,
    YHat (YHat),
    trainer,
    vegaLiteMonitor,
    _Batch,
  )

data Word2Vec windowSize vocabSize featureSize dtype device where
  Word2Vec ::
    { embed0 :: Embedding 'Nothing vocabSize featureSize 'Learned dtype device
    } ->
    Word2Vec windowSize vocabSize featureSize dtype device
  deriving stock (Show, Generic)
  deriving anyclass (Parameterized)

data Word2VecSpec windowSize vocabSize featureSize dtype device where
  Word2VecSpec :: Word2VecSpec windowSize vocabSize featureSize dtype device
  deriving stock (Show, Generic)

type WindowIsBalanced windowSize = (Mod windowSize 2 ~ 1, windowSize >= 1)

instance
  ( KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype,
    All KnownNat '[featureSize, vocabSize],
    StandardFloatingPointDTypeValidation
      device
      dtype,
    SumDTypeIsValid device dtype,
    SumDType dtype ~ dtype,
    WindowIsBalanced windowSize
  ) =>
  Randomizable
    (Word2VecSpec windowSize vocabSize featureSize dtype device)
    (Word2Vec windowSize vocabSize featureSize dtype device)
  where
  sample Word2VecSpec = do
    init' <- UnsafeMkTensor <$> kaimingUniform FanIn (LeakyRelu $ Prelude.sqrt (5.0 :: Float)) [fan_in, fan_out] -- FIXME: use a sensible initialization
    Word2Vec
      <$> sample (LearnedEmbeddingWithCustomInitSpec @ 'Nothing @vocabSize @featureSize @dtype @device init')
    where
      fan_in = natValI @featureSize
      fan_out = natValI @vocabSize

word2Vec ::
  forall batchSize windowSize vocabSize featureSize dtype device.
  ( All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    MeanDTypeValidation device dtype,
    AllDimsPositive '[windowSize -1, batchSize]
  ) =>
  Word2Vec windowSize vocabSize featureSize dtype device ->
  Bool ->
  [Tensor device 'D.Int64 '[batchSize]] ->
  IO (Tensor device dtype '[batchSize])
word2Vec Word2Vec {..} _stochastic input = do
  let (heads, i, tails) = getWindows' midpoint mempty (reshape @'[batchSize, featureSize] . forward embed0 <$> input)
      i' = toDynamic $ reshape @'[batchSize, featureSize] i
      heads' = cosineSimilarity' i' . toDynamic <$> heads -- FIXME: no typed cosineSimilarity in hasktorch currently
      tails' = cosineSimilarity' i' . toDynamic <$> tails
  pure . meanDim @0 $ powScalar (2 :: Int) (UnsafeMkTensor @_ @_ @'[windowSize -1, batchSize] (F.cat (Dim 0) (heads' <> tails')))
  where
    windowSize = natValI @windowSize
    midpoint = Prelude.floor (fromIntegral windowSize / 2.0 :: Double) + 1 :: Int
    getWindows' 0 h (x : xs) = (h, x, xs)
    getWindows' i h (x : xs) = getWindows' (i - 1) (x : h) xs
    getWindows' _ _ _ = error "word2Vec: getWindows': impossible. "

newtype TrainableWord2Vec (batchSize :: Nat) windowSize vocabSize featureSize dtype device = TrainableWord2Vec
  {unTrainableWord2Vec :: Word2Vec windowSize vocabSize featureSize dtype device}
  deriving stock (Generic)
  deriving newtype (Parameterized)

-- | This would be much simpler if we could just wrap a `Double` but we use this to pass the value to
-- the optimizer so it has to remain in torch-land
newtype Word2VecMetrics device dtype f = Word2VecMetrics
  { word2VecMetricCosine :: f (Tensor device dtype '[])
  }
  deriving stock (Generic)
  deriving anyclass (FunctorB, TraversableB, ApplicativeB, ConstraintsB)

deriving via
  (Barbie (Word2VecMetrics device dtype) f)
  instance
    AllBF Semigroup f (Word2VecMetrics device dtype) => Semigroup (Word2VecMetrics device dtype f)

deriving via
  (Barbie (Word2VecMetrics device dtype) f)
  instance
    AllBF Monoid f (Word2VecMetrics device dtype) => Monoid (Word2VecMetrics device dtype f)

instance
  ( All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    MeanDTypeValidation device dtype,
    'True ~ (1 <=? batchSize),
    AllDimsPositive '[windowSize -1, batchSize]
  ) =>
  HasMetricSpace (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device)
  where
  type
    Domain (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) =
      [Tensor device 'D.Int64 '[batchSize]]
  type
    Codomain (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) =
      Tensor device dtype '[batchSize]
  type
    MetricSpace f (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) =
      Word2VecMetrics device dtype f

  measure _ (YHat yhat) _ = Word2VecMetrics (Identity $ meanAll yhat)
  propagationStepM (TrainableWord2Vec m) s x = liftIO $ word2Vec m s x

newtype W2VTrainReport f = W2VTrainReport
  { w2VReportBatch :: f Integer
  }
  deriving stock (Generic)
  deriving anyclass (FunctorB, TraversableB, ApplicativeB, ConstraintsB)

instance
  ( All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    MeanDTypeValidation device dtype,
    KnownDevice device,
    'True ~ (1 <=? batchSize),
    AllDimsPositive '[windowSize -1, batchSize]
  ) =>
  HasTrainState (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) optim
  where
  data TrainState (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) optim = W2VState
    { w2VStateModel :: TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device,
      w2VStateBatch :: Integer,
      w2VStateOptim :: optim,
      w2VStateLr :: Tensor device 'D.Float '[],
      w2VStateMetrics :: Maybe (Word2VecMetrics device dtype Identity)
    }
  type TrainReport f (TrainableWord2Vec batchSize windowSize vocabSize featureSize dtype device) optim = W2VTrainReport f
  getModel = w2VStateModel
  getOptim = w2VStateOptim
  getTrainReport = W2VTrainReport . Identity . w2VStateBatch
  initTrainState m o = W2VState m 0 o 0.01 Nothing

type HasGradEtc model gradients =
  ( HasGrad
      ( HList
          ( Parameters
              model
          )
      )
      (HList gradients),
    HMap'
      ToDependent
      ( Parameters
          model
      )
      gradients,
    HMapM'
      IO
      MakeIndependent
      gradients
      ( Parameters
          model
      )
  )

-- | There's gotta be an alias for this whack-a-mole
type CanUseAdam batchSize windowSize vocabSize params featureSize device gradients =
  ( HZipWith3
      (AdamParameterUpdate device 'D.Float)
      gradients
      gradients
      gradients
      gradients,
    HMap'
      ZerosLike
      '[Parameter device 'D.Float '[vocabSize, featureSize]]
      gradients,
    HMap' AdamBiasAdjustment gradients gradients,
    HZipWith
      AdamMomentum2Update
      gradients
      '[Tensor device 'D.Float '[vocabSize, featureSize]]
      gradients,
    HZipWith
      AdamMomentum1Update
      gradients
      '[Tensor device 'D.Float '[vocabSize, featureSize]]
      gradients,
    HasGrad
      (HList params)
      (HList '[Tensor device 'D.Float '[vocabSize, featureSize]]),
    HasGrad
      ( HList
          ( Parameters
              ( TrainableWord2Vec
                  batchSize
                  windowSize
                  vocabSize
                  featureSize
                  'D.Float
                  device
              )
          )
      )
      (HList '[Tensor device 'D.Float '[vocabSize, featureSize]]),
    HMap'
      ZerosLike
      params
      '[Tensor device 'D.Float '[vocabSize, featureSize]],
    HMap'
      ToDependent
      ( Parameters
          ( TrainableWord2Vec
              batchSize
              windowSize
              vocabSize
              featureSize
              'D.Float
              device
          )
      )
      '[Tensor device 'D.Float '[vocabSize, featureSize]],
    HMapM'
      IO
      MakeIndependent
      '[Tensor device 'D.Float '[vocabSize, featureSize]]
      ( Parameters
          ( TrainableWord2Vec
              batchSize
              windowSize
              vocabSize
              featureSize
              'D.Float
              device
          )
      ),
    HMap'
      ToDependent
      '[Parameter device 'D.Float '[vocabSize, featureSize]]
      gradients,
    HMapM'
      IO
      MakeIndependent
      gradients
      '[Parameter device 'D.Float '[vocabSize, featureSize]],
    gradients
      ~ '[Tensor device 'D.Float '[vocabSize, featureSize]]
  )

instance
  ( SufficientlyCapable device,
    (1 <=? batchSize) ~ 'True,
    CanUseAdam batchSize windowSize vocabSize params featureSize device gradients,
    AllDimsPositive '[windowSize -1, batchSize],
    All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    HasTrainState (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) (Adam tensors),
    HasMetricSpace (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device),
    IsTrainable (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients,
    Optimizable (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients,
    HasGradEtc (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) gradients
  ) =>
  Trainable (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) (Adam tensors)
  where
  trainStepM (W2VState m b o lr _) (x, y) = do
    yhat <- liftIO $ stochasticStepM m x
    let metrics = measure Proxy yhat y
    (m', o') <- liftIO $ runStep m o (runIdentity $ word2VecMetricCosine metrics) lr
    pure $ W2VState m' (b + 1) o' lr (Just metrics)

-- | yeesh!
trainW2V ::
  forall batchSize windowSize featureSize vocabSize t m device params tensors gradients.
  ( IsStream t,
    MonadIO m,
    Monad (t m),
    MonadThrow m,
    CanUseAdam batchSize windowSize vocabSize params featureSize device gradients,
    AllDimsPositive '[windowSize -1, batchSize],
    MonadBaseControl IO m,
    (1 <=? batchSize) ~ 'True,
    All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    MeanDTypeValidation device 'D.Float,
    HasGradEtc (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) gradients,
    IsTrainable (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients
  ) =>
  TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device ->
  Adam tensors ->
  t
    m
    ( Domain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device)
    ) ->
  SerialT
    m
    ( Domain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device)
    ) ->
  t m (Report (Word2VecMetrics device 'D.Float Seq) (W2VTrainReport Maybe))
trainW2V = trainer @Seq @Maybe validToSingltonMonoid trainToSingltonMonoid
  where
    -- Should not do any monoidal aggregation here.  only passes batchwise metric.  Loose the stat monoids.
    validToSingltonMonoid :: Word2VecMetrics device 'D.Float Identity -> Word2VecMetrics device 'D.Float Seq
    validToSingltonMonoid (Word2VecMetrics nll) = Word2VecMetrics (Seq.singleton $ runIdentity nll)
    trainToSingltonMonoid = const (W2VTrainReport Nothing)

instance Monitorable [Report (Word2VecMetrics device 'D.Float Seq) (W2VTrainReport Maybe)] VegaLite where
  render as =
    toVegaLite $
      [ dataFromRows mempty dataRows
      ]
        <> lineFor 640 480 "Batch" "Wha?"
    where
      histoFor h w xName =
        let enc =
              encoding
                . position X [PName xName, PmType Quantitative, PBin []]
                . position Y [PAggregate Count, PmType Quantitative]
         in [ mark Bar [],
              enc [],
              height h,
              width w
            ]
      lineFor h w xName yName =
        let enc =
              encoding
                . position X [PName xName, PmType Temporal]
                . position Y [PName yName, PmType Quantitative]
         in [ mark Line [],
              enc [],
              height h,
              width w
            ]
      batches' = mapMaybe (w2VReportBatch <=< (^? _Batch)) as
      dataRows = (\f -> f mempty) $ foldl' go id batches'
        where
          go ls batchNumber =
            ls . dataRow [("Batch", Number (fromIntegral batchNumber))]

trainAndMonitor ::
  forall batchSize windowSize featureSize vocabSize m device params tensors gradients.
  ( MonadIO m,
    MonadThrow m,
    AllDimsPositive '[windowSize -1, batchSize],
    MonadBaseControl IO m,
    (1 <=? batchSize) ~ 'True,
    CanUseAdam batchSize windowSize vocabSize params featureSize device gradients,
    All KnownNat '[featureSize, vocabSize, batchSize, windowSize],
    MeanDTypeValidation device 'D.Float,
    HasGradEtc (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) gradients,
    IsTrainable (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients
  ) =>
  FilePath ->
  TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device ->
  Adam tensors ->
  SerialT
    m
    ( Domain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device)
    ) ->
  SerialT
    m
    ( Domain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Vec batchSize windowSize vocabSize featureSize 'D.Float device)
    ) ->
  m ()
trainAndMonitor fp m o tr vs = vegaLiteMonitor fp $ trainW2V m o tr vs
