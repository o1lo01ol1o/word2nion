{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Models.AutoEncoder where

import Control.Exception.Safe
  ( SomeException (..),
    try,
  )
import Control.Monad
  ( foldM,
    when,
  )
import Data.Proxy
import Foreign.ForeignPtr
import GHC.Generics
import GHC.TypeLits
import GHC.TypeLits.Extra
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
import qualified Torch.Serialize as D
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding
  ( dropout,
    linear,
  )
import Torch.Typed.NN
import Torch.Typed.Optim
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import qualified Torch.Typed.Vision as I
import Prelude hiding (tanh)


newtype ReconstructionError device dtype =
  ReconstructionError {unReconstructionError :: Tensor device dtype '[]}
  deriving stock (Show)
  deriving newtype (Num)

data
  AutoEncoderSpec
    (inputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  AutoEncoderSpec ::
    forall inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    {autoEncoderDropoutProbSpec :: Double} ->
    AutoEncoderSpec inputFeatures hiddenFeatures0 hiddenFeatures1 dtype
      device
  deriving (Show, Eq)

data
  AutoEncoder
    (inputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  AutoEncoder ::
    forall inputFeatures  hiddenFeatures0 hiddenFeatures1 dtype device.
    { autoEncoderLayer0 :: Linear inputFeatures hiddenFeatures0 dtype device,
      autoEncoderLayer1 :: Linear hiddenFeatures0 hiddenFeatures1 dtype device,
      autoEncoderLayer2 :: Linear hiddenFeatures1 inputFeatures dtype device,
      autoEncoderDropout :: Dropout
    } ->
    AutoEncoder inputFeatures hiddenFeatures0 hiddenFeatures1 dtype
      device
  deriving (Show, Generic)

instance
  ( KnownNat inputFeatures,
    KnownNat hiddenFeatures0,
    KnownNat hiddenFeatures1,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable (AutoEncoderSpec inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
    (AutoEncoder inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
  where
  sample AutoEncoderSpec {..} =
    AutoEncoder
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (DropoutSpec autoEncoderDropoutProbSpec)

autoEncoder ::
  forall
    batchSize
    inputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device.
  (StandardFloatingPointDTypeValidation device dtype) =>
  AutoEncoder inputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device ->
  Bool ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (Tensor device dtype '[batchSize, inputFeatures])
autoEncoder AutoEncoder {..} train input =
  return
    . linear autoEncoderLayer2
    =<< dropout autoEncoderDropout train
    . tanh
    . linear autoEncoderLayer1
    =<< dropout autoEncoderDropout train
    . tanh
    . linear autoEncoderLayer0
    =<< pure input