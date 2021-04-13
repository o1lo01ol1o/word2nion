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

{-# OPTIONS_GHC -Wno-redundant-constraints #-}
module Models.Word2FinVec where

--------------------------------------------------------------------------------
-- MLP for the morphism prediction
--------------------------------------------------------------------------------

import Data.Set (Set)
import qualified Data.Set as Set
import GHC.Generics (Generic)
import GHC.TypeNats (Div, type (<=?))
import GHC.TypeLits ( KnownNat, Nat, type (*), Mod, type (-) )
import Torch (asTensor)
import Torch.Functional (detach)
import qualified Torch.Functional as F
import Torch.Device as D

import Torch.DType as D
import Torch.Functional.Internal (maskedFillScalar, combinations)
import Torch.Typed
    ( Tensor(..),
      KnownDevice,
      Parameterized,
      RandDTypeIsValid,
      StandardFloatingPointDTypeValidation,
      HasForward(forward),
      Randomizable(..),
      natValI,
      argmax,
      divScalar,
      emptyLike,
      exp,
      log,
      relu,
      softmax,
      zerosLike,
      dropoutForward,
      type (>=),
      DimOutOfBoundCheck,
      StandardDTypeValidation,
      KeepOrDropDim(KeepDim),
      Dropout,
      DropoutSpec(DropoutSpec),
      Linear,
      LinearSpec(LinearSpec),
      Embedding(learnedEmbedWeights),
      EmbeddingSpec(LearnedEmbeddingWithRandomInitSpec),
      Parameter,
      KnownDType, reshape, All )
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

-- | Our Word2(the category of)Fin(ite)Vec(tor spaces) model.
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

-- | How to initialize the weights of the Word2FinVec model
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
    (Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device)
  where
  sample Word2FinVecSpec {..} =
    Word2FinVec
      <$> (learnedEmbedWeights <$> sample (LearnedEmbeddingWithRandomInitSpec @'Nothing @vocabSize @featureSize)) -- piggy-back off EmbeddingSpec for the weight matrix
      <*> pure morphs
      <*> sample mlpSpec
    where
      nMorphs = Set.size morphismTokenSetSpec
      morphs
        | nMorphs == (natValI @nMorphisms) = UnsafeMkTensor (F.oneHot (natValI @vocabSize) . asTensor $ Set.toList morphismTokenSetSpec)
        | otherwise = error "sample Word2FinVecSpec nMorphisms and provided morphismTokenSet are not the same size!"

-- | For simplicity, our context window needs to have an odd count of elements
type WindowIsBalanced windowSize = (Mod windowSize 2 ~ 1, windowSize >= 1)

-- | the style of output of the gumbel softmax function
data GumbelStyle
  = -- | output is a one-hot vector of values in {0,1}
    HardStyle
  | -- | output is the gumbel distribution
    SoftStyle
  deriving stock (Eq, Ord, Show, Bounded, Generic)

-- | see: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
-- and https://arxiv.org/abs/1611.01144
-- and https://arxiv.org/abs/1611.00712
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

-- | pairwise combinations of all values.  Flattens and uses `combinations` under the hood.
-- >>> let a = Torch.Typed.ones @'[3,5] @'D.Float @'(D.CPU, 0)
-- >>> Torch.Typed.shape $ pairsOfTokens a
-- [3,35,2]
pairsOfTokens ::
  forall batchSize windowSize np2 device dtype.
  ( All KnownNat '[batchSize, windowSize, np2],
    (Div (Div np2 batchSize) 2 * 2)
      ~ Div np2 batchSize,
    (1 <=? batchSize) ~ 'True,
    (batchSize * Div np2 batchSize) ~ np2,
    np2 ~ (((batchSize * windowSize) - 1) * (batchSize * windowSize))
  ) => Tensor device dtype '[batchSize, windowSize] -- ^ input
  ->
  Tensor device dtype '[batchSize, Div (Div np2 batchSize) 2, 2]
pairsOfTokens t = reshape . (UnsafeMkTensor @_ @_ @'[np2]) $ combinations (toDynamic $ reshape @'[batchSize * windowSize] t) 2 False

