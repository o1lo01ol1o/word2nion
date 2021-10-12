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
{-# LANGUAGE TemplateHaskell #-}
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
import Control.Lens.TH (makeLenses)
import qualified Data.Set as Set
import GHC.Generics (Generic)
import GHC.TypeNats (Div, type (<=?))
import GHC.TypeLits ( KnownNat, Nat, type (*), Mod, type (-) )
import Torch (asTensor)
import Torch.Functional (detach)
import qualified Torch.Functional as F
import Torch.Device as D

import Torch.DType as D
import Torch.Functional.Internal (maskedFillScalar, combinations, one_hot)
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
      KnownDType, reshape, All, dot, matmul, MatMul, toDependent, MatMulDTypeIsValid )
import Prelude hiding (exp, log)
import Control.Lens (Lens')
import Control.Monad.Reader

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
    { vocab :: Parameter device dtype '[vocabSize, featureSize], -- ^ The vocabulary
      morphismTokenSet :: Tensor device dtype '[nMorphisms, vocabSize], -- ^ The tokens we want to use as morphisms
      morphismMLP :: MLP (featureSize * 2) nMorphisms mlpHiddenSize dtype device, -- ^ The mlp for predicting morphisms from pairs of words
      morphismLittleTheta :: Parameter device dtype '[nMorphisms, 2 * featureSize], -- ^ The vector used to generate the matrix by which we functorially take tripls of (foo, morphism, bar) to a finite vector
      morphismBigTheta :: Parameter device dtype '[nMorphisms, featureSize] -- ^ The vector encoding the codomain of the universal map.
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

-- | Parmeters we may whish to vary accross batches
data BatchParam = BatchParam {_gumbelTau :: Float, _isStochastic :: Bool}
  deriving stock (Show, Eq, Generic)

makeLenses ''BatchParam

-- | Context for parmeters we may whish to vary accross batches

class HasBatchEnv e where
  batchEnvL :: Lens' e BatchParam

  batchEnvTao :: Lens' e Float
  batchEnvTao = batchEnvL . batchEnvTao

  batchEnvStochastic :: Lens' e Bool
  batchEnvStochastic = personEnvL . batchEnvStochastic

instance HasBatchEnv BatchParam where
  batchEnvL = id
  batchEnvTao f s = f (_gumbelTau s) <&> \a -> s {_gumbelTau = a}
  batchEnvStochastic f s = f (_isStochastic s) <&> \a -> s {_isStochastic = a}

askTao :: (MonadReader e m, HasBatchEnv e) => m Float
askTao = view batchEnvTao

askIsStochastic :: (MonadReader e m, HasBatchEnv e) => m Bool
askIsStochastic = view batchEnvStochastic

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
      <*> sample LinearSpec
      <*> sample LinearSpec
    where
      nMorphs = Set.size morphismTokenSetSpec
      morphs
        | nMorphs == (natValI @nMorphisms) = UnsafeMkTensor (F.oneHot (natValI @vocabSize) . asTensor $ Set.toList morphismTokenSetSpec)
        | otherwise = error "sample: Word2FinVecSpec: nMorphisms and provided morphismTokenSet are not the same size!"

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
-- Output shape is '[nPairs, pair]
-- >>> let a = Torch.Typed.ones @'[3,5] @'D.Float @'(D.CPU, 0)
-- >>> Torch.Typed.shape $ pairsOfTokens a
-- [105,2]
pairsOfTokens ::
  forall batchSize windowSize np2 device dtype.
  ( All KnownNat '[batchSize, windowSize, np2],
    (Div (Div np2 batchSize) 2 * 2)
      ~ Div np2 batchSize,
    (1 <=? batchSize) ~ 'True,
    (batchSize * Div np2 batchSize) ~ np2,
    (Div np2 2 * 2) ~ np2,
    np2 ~ (((batchSize * windowSize) - 1) * (batchSize * windowSize))
  ) =>
  -- | input
  Tensor device dtype '[batchSize, windowSize] -> 
  Tensor device dtype '[Div np2 2, 2]
pairsOfTokens t =
  reshape . (UnsafeMkTensor @_ @_ @'[np2]) $
    combinations (toDynamic $ reshape @'[batchSize * windowSize] t) 2 False
-- >>> let a = Torch.Typed.ones @'[3,5] @'D.Int32 @'(D.CPU, 0)
-- >>> Torch.Typed.shape $ oneHot @5 a
-- [3,5,5]
oneHot ::
  forall vocabSize batchSize nTokens  device.
  (KnownNat vocabSize) =>
  Tensor device 'D.Int32 '[batchSize, nTokens] ->
  Tensor device 'D.Float '[batchSize, nTokens, vocabSize]
oneHot t = UnsafeMkTensor $ one_hot dt vocabSize
  where
    dt = toDynamic t
    vocabSize = natValI @vocabSize

oneHotTokenToVocabEmbed ::
  forall vocabSize featureSize batchSize nTokens dtype device.
  ( '[batchSize, nTokens, featureSize]
      ~ MatMul '[vocabSize, featureSize] '[batchSize, nTokens, vocabSize],
    MatMulDTypeIsValid
      device
      dtype
  ) => Parameter device dtype '[vocabSize, featureSize] -- ^ Vocabulary matrix
  -> Tensor device dtype '[batchSize, nTokens, vocabSize] -- ^ sequence of one-hot vectors to embed
  ->
  Tensor device dtype '[batchSize, nTokens, featureSize]
oneHotTokenToVocabEmbed vocab =
  matmul @'[batchSize, nTokens, featureSize] @'[vocabSize, featureSize] @'[batchSize, nTokens, vocabSize]
    (toDependent vocab)

-- Should this take the outerproduct of [worda, mi] and then dot that with wordb?  subtract that from the corresponding morphismBigTheta?
morphismPredict :: forall windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device nPairs m. (MonadReader BatchParam m, MonadIO m) =>  Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device
   -> Tensor device dtype '[nPairs, 2, featureSize] 
   -> m (Tensor device dtype '[nPairs, nMorphisms])
morphismPredict Word2FinVec{..} pairs = do  
  tao <- askTao
  isStochastic <- askIsStochastic
  let pairs' = forward morphismMLP isStochastic $ reshape @'[nPairs, 2 * featureSize] pairs
  morph_y_hot <- gumbelSoftmax GumbelHardStyle tao pairs'
  let morph = matmul @'[nPairs, nMorphisms] @'[nMorphisms, featureSize] @'[nPairs, featureSize] morph morph_y_hot 

universalConstructionStep :: forall batchSize windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device. 
  Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device
  -> Tensor device 'D.Int32 '[batchSize, windowSize]
  -> Tensor device 'D.Int32 '[batchSize, windowSize]
  -> Tensor device dtype '[batchSize]
universalConstructionStep Word2FinVec{..} feats _negSamps = do
  let allPairsInBatch = oneHotTokenToVocabEmbed vocab . oneHot . pairsOfTokens $ feats
  pure allPairsInBatch

    
-- trainStep :: forall batchSize windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device. 
--   Word2FinVec windowSize vocabSize featureSize mlpHiddenSize nMorphisms dtype device
--   -> Tensor device D.Int32 '[batchSize, windowSize]
--   -> Tensor device dtype '[batchSize, windowSize]
--   -> Tensor device dtype '[batchSize]
-- trainStep = undefined 
