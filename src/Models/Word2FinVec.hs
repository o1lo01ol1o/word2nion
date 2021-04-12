{-# LANGUAGE AllowAmbiguousTypes #-}
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
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE NoStarIsType #-}

module Models.Word2FinVec where

--------------------------------------------------------------------------------
-- MLP for the morphism prediction
--------------------------------------------------------------------------------

import Data.Set (Set)
import qualified Data.Set as Set
import GHC.Generics (Generic)
import GHC.TypeLits
import Torch (asTensor)
import qualified Torch.DType as D
import Torch.Functional (detach, oneHot)
import qualified Torch.Functional as F
import Torch.Functional.Internal (maskedFillScalar)
import Torch.Typed
import Prelude hiding (exp, log)

-- | Spec for initializing the MLP
data
  MLPSpec
    (inputFeatures :: Nat)
    (nMorphisms :: Nat)
    (mlpHiddenSize :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  MLPSpec ::
    forall inputFeatures nMorphisms mlpHiddenSize dtype device.
    {mlpDropoutProbSpec :: Double} ->
    MLPSpec inputFeatures nMorphisms mlpHiddenSize dtype device
  deriving stock (Show, Eq)

-- | An MLP
data
  MLP
    (inputFeatures :: Nat)
    (nMorphisms :: Nat)
    (mlpHiddenSize :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  MLP ::
    forall inputFeatures nMorphisms mlpHiddenSize dtype device.
    { mlpLayer0 :: Linear inputFeatures mlpHiddenSize dtype device,
      mlpLayer1 :: Linear mlpHiddenSize mlpHiddenSize dtype device,
      mlpLayer2 :: Linear mlpHiddenSize nMorphisms dtype device,
      mlpDropout :: Dropout
    } ->
    MLP inputFeatures nMorphisms mlpHiddenSize dtype device
  deriving stock (Show, Generic)
  deriving anyclass (Parameterized)

-- | The forward propagation of an mlp
mlp ::
  forall
    inputFeatures
    batchSize
    nMorphisms
    mlpHiddenSize
    dtype
    device.
  (StandardFloatingPointDTypeValidation device dtype) =>
  -- | The mlp
  MLP inputFeatures nMorphisms mlpHiddenSize dtype device ->
  -- | True where when we should apply dropout (stochastic)
  Bool ->
  -- | input
  Tensor device dtype '[batchSize, inputFeatures] ->
  IO (Tensor device dtype '[batchSize, nMorphisms])
mlp MLP {..} stochastic input =
  forward mlpLayer2
    <$> ( dropoutForward mlpDropout stochastic
            . relu
            . forward mlpLayer1
            =<< dropoutForward mlpDropout stochastic
              . relu
              . forward mlpLayer0
            =<< pure input
        )

-- | How to initialize the MLP
instance
  ( KnownNat inputFeatures,
    KnownNat nMorphisms,
    KnownNat mlpHiddenSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (MLPSpec inputFeatures nMorphisms mlpHiddenSize dtype device)
    (MLP inputFeatures nMorphisms mlpHiddenSize dtype device)
  where
  sample MLPSpec {..} =
    MLP
      <$> sample LinearSpec
      <*> sample LinearSpec
      <*> sample LinearSpec
      <*> sample (DropoutSpec mlpDropoutProbSpec)

data Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device where
  Word2FinVec ::
    forall windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device.
    { vocab :: Parameter device dtype '[vocabSize, featureSize],
      morphismTokenSet :: Tensor device dtype '[nMorphisms, vocabSize],
      morphismMLP :: MLP (featureSize * 2) nMorphisms mlpHiddenSize dtype device
    } ->
    Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device
  deriving stock (Show, Generic)
  deriving anyclass (Parameterized)

data Word2FinVecSpec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device where
  Word2FinVecSpec ::
    { morphismTokenSetSpec :: Set Int,
      mlpSpec :: MLPSpec (featureSize * 2) nMorphisms mlpHiddenSize dtype device
    } ->
    Word2FinVecSpec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device
  deriving stock (Show, Eq, Generic)

-- | How to initialize the Word2FinVec
instance
  ( KnownNat nMorphisms,
    KnownNat vocabSize,
    KnownNat featureSize,
    KnownNat mlpHiddenSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (Word2FinVecSpec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device)
    (Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device) where 
      sample Word2FinVecSpec {..} = undefined


-- | For simplicity, our context window needs to have an odd count of elements
type WindowIsBalanced windowSize = (Mod windowSize 2 ~ 1, windowSize >= 1)

-- | the style of output of the gumbel softmax function
data GumbelStyle
  = -- | output is a one-hot vector of values in {0,1}
    HardStyle
  | -- | output is the gumbel distribution
    SoftStyle

-- | see: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
gumbelSoftmax ::
  forall dim shape dtype device.
  ( KnownNat dim,
    DimOutOfBoundCheck shape dim,
    KnownDevice device,
    KnownDType dtype,
    StandardDTypeValidation device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  -- | Hard or soft outputs
  GumbelStyle ->
  -- | Tau
  Float ->
  -- | input
  Tensor device dtype shape ->
  IO (Tensor device dtype shape)
gumbelSoftmax gs tau logits = case gs of
  SoftStyle -> softmax @dim <$> gumbelsIO
  HardStyle -> do
    ysoft' <- softmax @dim <$> gumbelsIO
    let indicies = argmax @dim @'KeepDim $ ysoft'
        yHard =
          UnsafeMkTensor @_ @_ @shape
            . (\t' -> maskedFillScalar t' (toDynamic indicies) 1)
            . toDynamic
            $ zerosLike logits
    detachedYSoft' <- UnsafeMkTensor @_ @_ @shape <$> detach (toDynamic ysoft')
    pure $ (yHard - detachedYSoft') + ysoft'
  where
    gumbelsIO :: IO (Tensor device dtype shape)
    gumbelsIO =
      divScalar tau . (logits -) . log . exp
        <$> emptyLike logits
