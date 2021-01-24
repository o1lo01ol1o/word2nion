{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
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

module Models.Word2Vec where

import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Mod, type (-))
import qualified Torch.DType as D
import Torch.Functional as F (Dim (Dim), cat, cosineSimilarity')
import Torch.Initializers
  ( FanMode (FanIn),
    NonLinearity (LeakyRelu),
    kaimingUniform,
  )
import Torch.Typed
  ( All,
    AllDimsPositive,
    Embedding,
    EmbeddingSpec (LearnedEmbeddingWithCustomInitSpec),
    EmbeddingType (Learned),
    HasForward (forward),
    KnownDType,
    KnownDevice,
    MeanDTypeValidation,
    RandDTypeIsValid,
    Randomizable (..),
    StandardFloatingPointDTypeValidation,
    SumDType,
    SumDTypeIsValid,
    Tensor (..),
    meanDim,
    natValI,
    powScalar,
    reshape,
    type (>=),
  )

data Word2Vec windowSize vocabSize featureSize dtype device where
  Word2Vec ::
    forall vocabSize windowSize featureSize dtype device.
    { embed0 :: Embedding 'Nothing vocabSize featureSize 'Learned dtype device
    } ->
    Word2Vec windowSize vocabSize featureSize dtype device
  deriving stock (Show, Generic)

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
