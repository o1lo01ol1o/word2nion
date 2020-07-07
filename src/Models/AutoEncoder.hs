{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
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

import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.NN as A
import Torch.Typed.Aux
import Torch.Typed.Factories
import Torch.Typed.Functional hiding
  ( linear,
  )
import Torch.Typed.NN ()
import Torch.Typed.NN
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (tanh)

newtype ReconstructionError device dtype = ReconstructionError {unReconstructionError :: Tensor device dtype '[]}
  deriving stock (Show)
  deriving newtype (Num)

data
  AutoEncoderSpec
    (inputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  AutoEncoderSpec ::
    forall inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    {autoEncoderDropoutProbSpec :: Double} ->
    AutoEncoderSpec
      inputFeatures
      hiddenFeatures0
      hiddenFeatures1
      dtype
      device
  deriving stock (Show, Eq)

data
  AutoEncoder
    (inputFeatures :: Nat)
    (hiddenFeatures0 :: Nat)
    (hiddenFeatures1 :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  AutoEncoder ::
    forall inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device.
    { autoEncoderLayer0 :: Linear inputFeatures hiddenFeatures0 dtype device,
      autoEncoderLayer1 :: Linear hiddenFeatures0 hiddenFeatures1 dtype device,
      autoEncoderLayer2 :: Linear hiddenFeatures1 inputFeatures dtype device,
      autoEncoderDropout :: Dropout
    } ->
    AutoEncoder
      inputFeatures
      hiddenFeatures0
      hiddenFeatures1
      dtype
      device
  deriving stock (Show, Generic)

instance
  ( KnownNat inputFeatures,
    KnownNat hiddenFeatures0,
    KnownNat hiddenFeatures1,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (AutoEncoderSpec inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device)
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
  AutoEncoder
    inputFeatures
    hiddenFeatures0
    hiddenFeatures1
    dtype
    device ->
  Bool ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (Tensor device dtype '[batchSize, inputFeatures])
autoEncoder AutoEncoder {..} doStochastic input =
  return
    . linearForward autoEncoderLayer2
    =<< dropoutForward autoEncoderDropout doStochastic
      . tanh
      . linearForward autoEncoderLayer1
    =<< dropoutForward autoEncoderDropout doStochastic
      . tanh
      . linearForward autoEncoderLayer0
    =<< pure input

-- | Approximate the upper bound of the AIC via mean-squared reconstruction error
aicMSRE ::
  (StandardFloatingPointDTypeValidation device dtype
  , KnownDevice device
  , MeanDTypeValidation device dtype
  , (1 <=? inputFeatures) ~ 'True
  , (1 <=? batchSize) ~ 'True
  ) =>
  AutoEncoder inputFeatures hiddenFeatures0 hiddenFeatures1 dtype device ->
  Bool ->
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (ReconstructionError device dtype)
aicMSRE ae doStochastic input = do
  r <- autoEncoder ae doStochastic input
  pure . ReconstructionError . meanAll $ pow (2 :: Int) (input - r)
