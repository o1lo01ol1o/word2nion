{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module Torch.Typed.Entropy where

import Fcf.Core
import Fcf.Data.List
import GHC.TypeLits
import Torch.Typed.Aux
import Torch.Typed.Functional hiding
  ( dropout,
    linear,
  )
import Torch.Typed.Tensor
import Prelude hiding (log, sum, tanh)

newtype Entropy device dtype = Entropy {unEntropy :: Tensor device dtype '[]}
  deriving stock (Show)
  deriving newtype (Num)

type HasEntropy shape device dtype =
  ( StandardFloatingPointDTypeValidation device dtype,
    KnownDType dtype,
    All KnownNat shape,
    MeanDTypeValidation device dtype,
    SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype
  )

-- | Shannon entropy of a probability distibution is just -Σᵢ₋ⱼ xᵢ * logxᵢ
-- N.B. This function assumes that the inputs are already in the range [0,1]
entropy ::
  forall lastDim shape device dtype.
  ( HasEntropy shape device dtype,
    KnownNat lastDim,
    AllDimsPositive (Eval (Take lastDim shape)),
    DropValue shape lastDim ~ Eval (Take lastDim shape)
  ) =>
  Tensor device dtype shape ->
  Entropy device dtype
entropy = Entropy . meanAll . mulScalar (-1.0 :: Float) . sumDim @lastDim . log
