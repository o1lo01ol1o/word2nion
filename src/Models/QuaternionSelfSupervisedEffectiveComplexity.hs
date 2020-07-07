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

-- | Defines a model that seeks to maximize the expected complexity of a sequence
-- of tokens sourced from real-world copora and minimize the expected complexity
-- of tokens from artificial (noise) corpora.
module Models.QuaternionSelfSupervisedEffectiveComplexity where

import Barbies
import Control.Exception.Safe (SomeException (..), try)
import Control.Monad (foldM, when)
import Control.Monad.Catch
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.Trans.Control hiding (embed)
import Data.Bifunctor (second)
import Data.Functor.Const
import Data.Functor.Identity
import Data.Functor.Product (Product (..))
import qualified Data.Monoid.Statistics as StatM
import Data.Proxy
import Data.Sequence (Seq)
import qualified Data.Sequence as Seq
import Fcf.Core
import Fcf.Data.List
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import Generic.Data (Generically (..))
import Graphics.Vega.VegaLite hiding (Identity)
import Models.AutoEncoder
import Streamly
import Streamly
import qualified Streamly.Prelude as S
import System.Environment
import System.IO.Unsafe
import System.Random
import qualified Torch.Autograd as A
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.NN as A
import Torch.Streamly.Dataloader
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Aux
import Torch.Typed.Entropy
import Torch.Typed.Factories
import Torch.Typed.Functional hiding
  ( conv2d,
    linear,
  )
import Torch.Typed.NN
import Torch.Typed.Parameter
import Torch.Typed.Quaternion
import Torch.Typed.Tensor
import Trainer
import Prelude hiding (abs, tanh)

-- | Quaternion model consisting of an embedding matrix of size (vocabSize, featureSize)
-- and an autoencoder used to learn a compression of the "articulation model" created
-- by the successive hamilton products of tokens.  This compression approximates the
-- algorithmic information content (AIC) of the "articulation model".  It's an open question
-- as to the degree to which the "articulation model" can be both a model and the probability distrobution
-- over the data ie, the next token.  In Gell-Mann, they are separate, but the distribution forms a lower bound on the
-- complexity of a modle since any model could just be a constant distibution.
data Word2QuatEffComplex vocabSize featureSize dtype device where
  Word2QuatEffComplex ::
    forall vocabSize featureSize dtype device.
    { embed0 :: Embedding 'Nothing vocabSize featureSize 'Learned dtype device,
      ae0 ::
        AutoEncoder
          featureSize
          (Div featureSize 4)
          (Div featureSize 4)
          dtype
          device
    } ->
    Word2QuatEffComplex vocabSize featureSize dtype device
  deriving stock (Show, Generic)

-- | The initialization spec takes an `alpha` parameter to control the scaling of
-- of the two loss functions we want to use (the AIC via reconstruction error
-- and the shannon entropy / cosine distance between "articulation model" and subsequent token.)
-- It also take as dropout parameter to hand off to the autoencoder's initialization.
data Word2QuatEffComplexSpec vocabSize featureSize dtype device where
  Word2QuatEffComplexSpec :: {w2qecAlpha :: Double, w2qecDropout :: Double} -> Word2QuatEffComplexSpec vocabSize featureSize dtype device
  deriving stock (Show, Generic)

-- | Initialize with quaternions and a random Autoeencoder
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
    (Word2QuatEffComplexSpec vocabSize featureSize dtype device)
    (Word2QuatEffComplex vocabSize featureSize dtype device)
  where
  sample Word2QuatEffComplexSpec {..} = do
    init' <- catQuaternions @'[vocabSize, (Div featureSize 4)] <$> initialize @vocabSize @(Div featureSize 4)
    Word2QuatEffComplex
      <$> A.sample (LearnedEmbeddingWithCustomInitSpec @'Nothing @vocabSize @featureSize @dtype @device init')
      <*> A.sample (AutoEncoderSpec w2qecDropout)

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
-- approximates KC. Should the AIC be estimated on the quaternions of q1 and q2?  I suppose it needs to since
-- the shannon entropy depends on both.
word2QuatEffComplex ::
  forall batchSize vocabSize featureSize dim dtype device.
  ( HasQuaternionComponents '[batchSize, featureSize] dim featureSize device dtype,
    'True ~ (1 <=? (Div featureSize 4)),
    'True ~ (1 <=? batchSize),
    KnownDType dtype,
    SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype,
    CanNormalize '[batchSize, featureSize] device dtype,
    MeanDTypeValidation device dtype
  ) =>
  Word2QuatEffComplex vocabSize featureSize dtype device ->
  Bool ->
  [Tensor device 'D.Int64 '[batchSize]] ->
  IO (Tensor device dtype '[batchSize, featureSize], (Entropy device dtype, ReconstructionError device dtype))
word2QuatEffComplex Word2QuatEffComplex {..} stochastic input = do
  let e = reshape @'[batchSize, featureSize] . forward embed0 <$> input
  let r' = hamiltonReduce Nothing e
  let is = meanOverSequence $ (\(q1, q2) -> unEntropy . entropy @1 $ approxDistances q1 q2) <$> byTwos r'
  aic <- meanOverSequence <$> traverse (fmap unReconstructionError . aicMSRE ae0 stochastic) r'
  pure (last e, (Entropy is, ReconstructionError aic))
  where
    meanOverSequence :: [Tensor device dtype shape] -> Tensor device dtype shape
    meanOverSequence xs = mulScalar ((1.0 / (fromIntegral $ length xs)) :: Double) (sum xs)
