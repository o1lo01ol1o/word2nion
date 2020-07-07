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
--
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
--
data Word2QuatEffComplexSpec vocabSize featureSize dtype device where
  Word2QuatEffComplexSpec :: {w2qecAlpha :: Double, w2qecDropout :: Double} -> Word2QuatEffComplexSpec vocabSize featureSize dtype device
  deriving stock (Show, Generic)

-- | Initialize with quaternions and a random Autoeencoder
--
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

-- | word2Quat but caclulates the loss based on the entropy of the next word and the current "articulation model"
-- (the last hamiton product) and the AIC (via a `ReconstructionError`) of "articulation model".  Each term is averaged
-- over the sequence of tokens.
--
-- TODO: There are a few ways to figure out how to develop this.  Probably these should be pursude after
-- a simpler analyssi  of the quaternion-based approach vs word2vec.
-- Two thoughts:  currently, AIC and SE are caclulated on the same data (modulo a linear transformation)
-- In Gell-Mann, EC is the AIC of "model" that issues a probability distribution over the data.  As such, it may
-- be worth it to add an MLP that takes the current state and predicts the next token (either via) emitting the raw token
-- and taking the distance or by using a hierarchical softmax over the vocab.  This is closer to traditional language modeling
-- but motivates using the AIC on a model instead of conflating the "model" with the currnet state of the reading as well as the 
-- asking the reading to anticipate the next token.
-- A more complex option would be to try to maximize the integrated (effective) complexity on the vocabulary itself.
--
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
