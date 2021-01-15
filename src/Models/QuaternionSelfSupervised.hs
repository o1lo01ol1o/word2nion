{-# LANGUAGE ConstraintKinds #-}
-- {-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
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

-- | Defines a model similar to the one defined in the word2vec papers
-- except it is represented by quaternions and it can support aribitrary lenghts
-- of sequences.
module Models.QuaternionSelfSupervised where

import Barbies
import Control.Lens ((^.), (^?), _1, _2)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Control (MonadBaseControl)
import Data.Functor.Identity (Identity (..))
import Data.Functor.Product (Product (..))
import qualified Data.Monoid.Statistics as StatM
import Data.Proxy (Proxy (Proxy))
import Data.Sequence (Seq)
import qualified Data.Sequence as Seq
import GHC.Generics (Generic, M1 (M1))
import GHC.TypeLits (Div, KnownNat, Nat, type (+), type (<=?))
import Graphics.Vega.VegaLite
  ( DataRow,
    DataValue (Number),
    FieldName,
    Format,
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
    vConcat,
    width,
  )
import Streamly (IsStream, Semigroup, SerialT)
import Streamly.Prelude as S (foldl', mapMaybe)
import qualified Torch.DType as D
import qualified Torch.NN as A
import Torch.Streamly.Dataloader (byTwos)
import Torch.Typed.Aux (StandardFloatingPointDTypeValidation)
import Torch.Typed.Factories (RandDTypeIsValid)
import Torch.Typed.Functional
  ( MeanDTypeValidation,
    SumDType,
    SumDTypeIsValid,
    meanAll,
    meanDim,
    mulScalar,
    powScalar,
  )
import Torch.Typed.NN
  ( Embedding,
    EmbeddingSpec (LearnedEmbeddingWithCustomInitSpec),
    EmbeddingType (Learned),
    HasForward (forward),
  )
import Torch.Typed.Optim (Adam, runStep)
import Torch.Typed.Parameter (Randomizable (sample))
import Torch.Typed.Quaternion
  ( CanNormalize,
    DivisionProofs,
    HasQuaternionComponents,
    HasQuaternions,
    approxDistances,
    catQuaternions,
    hamiltonReduce,
    initialize,
  )
import Torch.Typed.Tensor
  ( All,
    CheckBroadcast,
    ComputeBroadcast,
    KnownDType,
    KnownDevice,
    Tensor,
    reshape,
    toFloat,
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
import Prelude hiding (abs, tanh)

-- | (Vanilla) quaternion model is just an embedding matrix of size (vocabSize, featureSize)
data Word2Quat vocabSize featureSize dtype device where
  Word2Quat ::
    forall vocabSize featureSize dtype device.
    { embed0 :: Embedding 'Nothing vocabSize featureSize 'Learned dtype device
    --   fc0 :: Linear featureSize 1 dtype device,
    --   w2qDropout :: Dropout
    } ->
    Word2Quat vocabSize featureSize dtype device
  deriving stock (Show, Generic)

-- | The initialization spec is boring in this case
data Word2QuatSpec vocabSize featureSize dtype device where
  Word2QuatSpec :: Word2QuatSpec vocabSize featureSize dtype device
  deriving stock (Show, Generic)

-- | ... we just initialize the matrix with random quaternions.
instance
  ( KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype,
    HasQuaternions featureSize,
    DivisionProofs featureSize 4,
    All KnownNat '[featureSize, vocabSize],
    StandardFloatingPointDTypeValidation
      device
      dtype,
    SumDTypeIsValid device dtype,
    SumDType dtype ~ dtype
  ) =>
  A.Randomizable
    (Word2QuatSpec vocabSize featureSize dtype device)
    (Word2Quat vocabSize featureSize dtype device)
  where
  sample Word2QuatSpec = do
    init' <- catQuaternions @'[vocabSize, Div featureSize 4] <$> initialize @vocabSize @(Div featureSize 4)
    Word2Quat
      <$> A.sample (LearnedEmbeddingWithCustomInitSpec @ 'Nothing @vocabSize @featureSize @dtype @device init')

type CanPropagate batchSize vocabSize featureSize dim dtype device =
  ( HasQuaternionComponents '[batchSize, featureSize] dim featureSize device dtype,
    'True ~ (1 <=? Div featureSize 4),
    'True ~ (1 <=? batchSize),
    CanNormalize '[batchSize, featureSize] device dtype,
    MeanDTypeValidation device dtype
  )

-- | The forward propagation function.  It takes a model, a boolean designating dropout use (unused), and input data
-- in the form of a list of tokens in a sequence (of `batchSize` individual sequences). output is a tensor of mean squared
-- errors (where error is cacluated as the average squared distance between all intermediary states and next tokens for all batches) over each sequence.
--
-- Interesting idea: in word2vec, we ask the word reps to be similar subject to their co-occurance
-- Instead of asking them to be similar, what if we ask them to maximize their effective complexity?
-- We can take the output of `approxDistances` to be a probability distibution for shannon entropy:
-- Ie, a vector populated by probabilities.  In this case, the entropy is 0 when q1 and q2 are the same.
-- intuitively, this makes sense: if the total of all previous words in a text perfectly imply the next word,
-- there's zero uncertainty between them.  Conversly, if q2 comes out of nowhere, thre's higher uncertainty.
-- The AIC (Komogorov Complexity) can be estimated by learning an autoencoder that tries to compress the quaternions
-- themselves.  This will output a reconstruction error that is a metric of relative uncompressability and
-- approximates KC. Shhould the AIC be estimated on the quaternions of q1 and q2?  I suppose it needs to since
-- the shannon entropy depends on both.
word2Quat ::
  forall batchSize vocabSize featureSize dim dtype device.
  ( CanPropagate batchSize vocabSize featureSize dim dtype device
  ) =>
  Word2Quat vocabSize featureSize dtype device ->
  Bool ->
  [Tensor device 'D.Int64 '[batchSize]] ->
  IO (Tensor device dtype '[batchSize])
word2Quat Word2Quat {..} _stochastic input = do
  let e = reshape @'[batchSize, featureSize] . forward embed0 <$> input
  let r' = hamiltonReduce Nothing e
  pure . meanDim @1 . mulScalar (1.0 / fromIntegral (length r') :: Double)
    . sum
    $ (\(q1, q2) -> powScalar (2 :: Int) $ approxDistances q1 q2) <$> byTwos r'

-- =============================================================================
-- Training intances
-- =============================================================================

-- | The above Word2Quat datatype provides the needed model definition, but to train
-- we need one more parameter to make sure all our operations are kosher: batchSize.
-- We use a newtype wrapper with a phantom type called batchSize to do this.
newtype TrainableWord2Quat (batchSize :: Nat) vocabSize featureSize dtype device = TrainableWord2Quat {unTrainableWord2Quat :: Word2Quat vocabSize featureSize dtype device}

-- | The metrics we're going to want to keep track of.
newtype Word2QuatMetrics device dtype f = Word2QuatMetrics
  { word2QuatMetricCosine :: f (Tensor device dtype '[])
  }
  deriving stock (Generic)
  deriving anyclass (FunctorB, TraversableB, ApplicativeB, ConstraintsB)

deriving via
  (Barbie (Word2QuatMetrics device dtype) f)
  instance
    AllBF Semigroup f (Word2QuatMetrics device dtype) => Semigroup (Word2QuatMetrics device dtype f)

deriving via
  (Barbie (Word2QuatMetrics device dtype) f)
  instance
    AllBF Monoid f (Word2QuatMetrics device dtype) => Monoid (Word2QuatMetrics device dtype f)

-- | The needed types and methods to propagate inputs and score the outputs of the model
-- yeesh.
instance
  (CanPropagate batchSize vocabSize featureSize dim dtype device) =>
  HasMetricSpace (TrainableWord2Quat batchSize vocabSize featureSize dtype device)
  where
  type
    Domain (TrainableWord2Quat batchSize vocabSize featureSize dtype device) =
      [Tensor device 'D.Int64 '[batchSize]]
  type
    Codomain (TrainableWord2Quat batchSize vocabSize featureSize dtype device) =
      Tensor device dtype '[batchSize]
  type
    MetricSpace f (TrainableWord2Quat batchSize vocabSize featureSize dtype device) =
      Word2QuatMetrics device dtype f

  measure _ (YHat yhat) _ = Word2QuatMetrics (Identity $ meanAll yhat)
  propagationStepM (TrainableWord2Quat m) s x = liftIO $ word2Quat m s x

-- | The Training report differes in scope from the model metrics
-- since it has access to more data, such as the current batch number
-- (and, eg, gradient statistics)
newtype W2QTrainReport f = W2QTrainReport
  { w2QReportBatch :: f Integer
  }
  deriving stock (Generic)
  deriving anyclass (FunctorB, TraversableB, ApplicativeB, ConstraintsB)

-- | This is basically boilerplate to get access to different parts of the training state and
-- formalize what the training state consists in.
instance
  (CanPropagate batchSize vocabSize featureSize dim dtype device) =>
  HasTrainState (TrainableWord2Quat batchSize vocabSize featureSize dtype device) optim
  where
  data TrainState (TrainableWord2Quat batchSize vocabSize featureSize dtype device) optim = W2QState
    { w2QStateModel :: TrainableWord2Quat batchSize vocabSize featureSize dtype device,
      w2QStateBatch :: Integer,
      w2QStateOptim :: optim,
      w2QStateLr :: Tensor device 'D.Float '[],
      w2QStateMetrics :: Maybe (Word2QuatMetrics device dtype Identity)
    }
  type TrainReport f (TrainableWord2Quat batchSize vocabSize featureSize dtype device) optim = W2QTrainReport f
  getModel = w2QStateModel
  getOptim = w2QStateOptim
  getTrainReport = W2QTrainReport . Identity . w2QStateBatch
  initTrainState m o = W2QState m 0 o 0.01 Nothing

-- | The actual training step.  Takes data and target in, batchwise, props the model,
-- scores the output, and updates the parameters given an optimizer.
instance
  ( CanPropagate batchSize vocabSize featureSize dim 'D.Float device,
    SufficientlyCapable device,
    HasTrainState (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) (Adam tensors),
    HasMetricSpace (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device),
    IsTrainable (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients,
    Optimizable (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients
  ) =>
  Trainable (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) (Adam tensors)
  where
  trainStepM (W2QState m b o lr _) (x, y) = do
    yhat <- liftIO $ stochasticStepM m x
    let metrics = measure Proxy yhat y
    (m', o') <- liftIO $ runStep m o (runIdentity $ word2QuatMetricCosine metrics) lr
    pure $ W2QState m' (b + 1) o' lr (Just metrics)

type SummaryStats = (Mean `Product` Variance) `Product` Seq

getMeanPart :: SummaryStats a -> Mean a
getMeanPart ss = ss ^. _1 . _1

getVarPart :: SummaryStats a -> Variance a
getVarPart ss = ss ^. _1 . _2

getSeqPart :: SummaryStats a -> Seq a
getSeqPart ss = ss ^. _2

-- | yeesh!
trainEcNN ::
  forall batchSize featureSize vocabSize t m device params tensors gradients.
  ( IsStream t,
    MonadIO m,
    Monad (t m),
    MonadThrow m,
    MonadBaseControl IO m,
    KnownNat batchSize,
    (1 <=? batchSize) ~ 'True,
    (1 <=? Div featureSize 4) ~ 'True,
    ( (Div featureSize 2 + Div featureSize 4)
        <=? featureSize
    )
      ~ 'True,
    CheckBroadcast
      '[batchSize, Div featureSize 4]
      '[batchSize, featureSize]
      ( ComputeBroadcast
          '[Div featureSize 4, batchSize]
          '[featureSize, batchSize]
      )
      ~ '[batchSize, featureSize],
    KnownNat featureSize,
    DivisionProofs featureSize 4,
    MeanDTypeValidation device 'D.Float,
    IsTrainable (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients,
    StatM.StatMonoid (Product Mean Variance (Tensor device 'D.Float '[])) Float,
    Monoid (Product (Product Mean Variance) Seq (Tensor device 'D.Float '[]))
  ) =>
  TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device ->
  Adam tensors ->
  t
    m
    ( Domain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device)
    ) ->
  SerialT
    m
    ( Domain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device)
    ) ->
  t m (Report (Word2QuatMetrics device 'D.Float SummaryStats) (W2QTrainReport Maybe))
trainEcNN = trainer @SummaryStats @Maybe validToSingltonMonoid trainToSingltonMonoid
  where
    fromTensor a = Pair (StatM.singletonMonoid . toFloat . runIdentity $ a) (Seq.singleton $ runIdentity a)
    validToSingltonMonoid :: Word2QuatMetrics device 'D.Float Identity -> Word2QuatMetrics device 'D.Float SummaryStats
    validToSingltonMonoid (Word2QuatMetrics nll) = Word2QuatMetrics (fromTensor nll)
    trainToSingltonMonoid = const (W2QTrainReport Nothing)

vegaLinePlot :: FieldName -> FieldName -> [Format] -> [DataRow] -> Double -> Double -> VegaLite
vegaLinePlot xName yName lineFormat lineData h w =
  let enc =
        encoding
          . position X [PName xName, PmType Temporal]
          . position Y [PName yName, PmType Quantitative]
   in toVegaLite
        [ dataFromRows lineFormat lineData,
          mark Line [],
          enc [],
          height h,
          width w
        ]

vegaLitesimpleHistogram :: FieldName -> [Format] -> [DataRow] -> Double -> Double -> VegaLite
vegaLitesimpleHistogram xName lineFormat lineData h w =
  let histo =
        encoding
          . position X [PName xName, PmType Quantitative, PBin []]
          . position Y [PAggregate Count, PmType Quantitative]
      ln =
        encoding
          . position X [PName "Batch", PmType Temporal]
          . position Y [PName xName, PmType Quantitative]
   in toVegaLite
        [ dataFromRows lineFormat lineData,
          mark Bar [],
          histo [],
          height h,
          width w
        ]

instance Monitorable [Report (Word2QuatMetrics device 'D.Float SummaryStats) (W2QTrainReport Maybe)] VegaLite where
  render as =
    toVegaLite
      [ dataFromRows mempty (dataRows mempty),
        vConcat [lineFor 640 480 "Nll-Mean" "Batch", lineFor 640 480 "Nll-Std" "Batch"]
      ]
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
      batches' = mapMaybe (^? _Batch . w2QReportBatch) as
      dataRows :: [DataRow] -> [DataRow]
      dataRows = foldl' go id (zip [1 ..] batches')
        where
          go ls (b, vs) =
            let stdB = StatM.calcStddev $ getStdPart v
                meanB = StatM.calcMean $ getMeanPart v
             in ls . dataRow [("Batch", Number b), ("Nll-Std", Number stdB), ("Nll-Mean", Number meanB)]

trainAndMonitor ::
  forall batchSize featureSize vocabSize m device params tensors gradients.
  ( MonadIO m,
    MonadThrow m,
    MonadBaseControl IO m,
    KnownNat batchSize,
    (1 <=? batchSize) ~ 'True,
    (1 <=? Div featureSize 4) ~ 'True,
    ( (Div featureSize 2 + Div featureSize 4)
        <=? featureSize
    )
      ~ 'True,
    CheckBroadcast
      '[batchSize, Div featureSize 4]
      '[batchSize, featureSize]
      ( ComputeBroadcast
          '[Div featureSize 4, batchSize]
          '[featureSize, batchSize]
      )
      ~ '[batchSize, featureSize],
    KnownNat featureSize,
    DivisionProofs featureSize 4,
    MeanDTypeValidation device 'D.Float,
    IsTrainable (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device) params tensors device (Adam tensors) gradients,
    StatM.StatMonoid (Product Mean Variance (Tensor device 'D.Float '[])) Float,
    Monoid (Product (Product Mean Variance) Seq (Tensor device 'D.Float '[]))
  ) =>
  FilePath ->
  TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device ->
  Adam tensors ->
  SerialT
    m
    ( Domain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device)
    ) ->
  SerialT
    m
    ( Domain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device),
      Codomain (TrainableWord2Quat batchSize vocabSize featureSize 'D.Float device)
    ) ->
  m ()
trainAndMonitor fp m o tr vs = vegaLiteMonitor fp $ trainEcNN m o tr vs
