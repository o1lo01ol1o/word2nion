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

module Models.QuaternionSelfSupervised where

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
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
import Generic.Data (Generically (..))
import Graphics.Vega.VegaLite hiding (Identity)
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
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding
  ( conv2d,
    linear,
  )
import Torch.Typed.NN
import Torch.Typed.NN.Dropout
import Torch.Typed.NN.Linear
import Torch.Typed.NN.Sparse
import Torch.Typed.Parameter
import Torch.Typed.Quaternion
import Torch.Typed.Tensor
import Trainer
import Prelude hiding (abs, tanh)

data Word2Quat vocabSize featureSize dtype device where
  Word2Quat ::
    forall vocabSize featureSize dtype device.
    { embed0 :: Embedding 'Nothing vocabSize featureSize 'Learned dtype device,
      fc0 :: Linear featureSize 1 dtype device,
      w2qDropout :: Dropout
    } ->
    Word2Quat vocabSize featureSize dtype device
  deriving stock (Show, Generic)

data Word2QuatSpec vocabSize featureSize dtype device where
  Word2QuatSpec :: {w2qDropoutProbSpec :: Double} -> Word2QuatSpec vocabSize featureSize dtype device
  deriving stock (Show, Generic)

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
  sample Word2QuatSpec {..} = do
    init' <- catQuaternions @'[vocabSize, (Div featureSize 4)] <$> initialize @vocabSize @(Div featureSize 4)
    Word2Quat
      <$> A.sample (LearnedEmbeddingWithCustomInitSpec @'Nothing @vocabSize @featureSize @dtype @device init')
      <*> A.sample (LinearSpec)
      <*> A.sample (DropoutSpec w2qDropoutProbSpec)

word2Quat ::
  forall batchSize vocabSize featureSize dtype device.
  ( StandardFloatingPointDTypeValidation
      device
      dtype
  ) =>
  Word2Quat vocabSize featureSize dtype device ->
  Bool ->
  [Tensor device 'D.Int64 '[batchSize]] ->
  IO (Tensor device dtype '[batchSize, 1])
word2Quat Word2Quat {..} _stochastic input = do
  let e = reshape @'[batchSize, featureSize] . forward embed0 <$> input
  let r' = hamiltonReduce e
  pure . tanh . forward fc0 $ r'
  where
    hamiltonReduce :: [Tensor device dtype '[batchSize, featureSize]] -> Tensor device dtype '[batchSize, featureSize]
    hamiltonReduce [] = error "hamiltonReduce: impossible"
    hamiltonReduce (x : []) = x
    hamiltonReduce (x : y : xs) = hamiltonReduce $ (hamilton x y) : xs
