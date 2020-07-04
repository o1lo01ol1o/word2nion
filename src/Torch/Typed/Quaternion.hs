{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

-- | Module defining operations on quaternions.
-- Ported from https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py
module Torch.Typed.Quaternion
  ( (⦿),
    hamilton,
    initialize,
    -- normalize,
    Quaternions (..),
    catQuaternions,
    HasQuaternions,
    NQuaternions,
    DivisionProofs,
  )
where

import Fcf.Core
import Fcf.Data.List
import Fcf.Utils
import GHC.Generics (Generic)
import GHC.TypeLits
import GHC.TypeNats ()
import Torch.Initializers
  ( FanMode (..),
    NonLinearity (..),
    kaimingUniform,
  )
import Torch.TensorFactories (randIO')
import Torch.Typed
import qualified Prelude as P (sqrt)
import Prelude hiding (cos, pi, sin, sqrt, sum)

type family NQuaternions (features :: Nat) where
  NQuaternions n = Div n 4

type HasQuaternions f = ((f `Mod` 4) ~ 0)

type DivisionProofs n d =
  ( n `Div` d <= n,
    Div n d + Div n d <= n,
    Div n (d `Div` 2) + Div n d <= n,
    (n - Div n d) + Div n d <= n,
    (Div n d * d)
      ~ ((n - Div n d) + Div n d)
  )

type HasQuaternionComponents shape dim featureSize device dtype =
  ( Narrow shape dim 0 (Div featureSize 4) ~ Eval (ApplyToLast QuaternionComponent shape),
    Narrow shape dim (featureSize `Div` 4) (Div featureSize 4) ~ Eval (ApplyToLast QuaternionComponent shape),
    Narrow shape dim (featureSize `Div` 2) (Div featureSize 4) ~ Eval (ApplyToLast QuaternionComponent shape),
    Narrow shape dim (featureSize - (featureSize `Div` 4)) (Div featureSize 4) ~ Eval (ApplyToLast QuaternionComponent shape),
    '(shape, dtype, device)
      ~ Cat
          (Eval (Length shape))
          '[ Tensor device dtype (Eval (ApplyToLast QuaternionComponent shape)),
             Tensor device dtype (Eval (ApplyToLast QuaternionComponent shape)),
             Tensor device dtype (Eval (ApplyToLast QuaternionComponent shape)),
             Tensor device dtype (Eval (ApplyToLast QuaternionComponent shape))
           ],
    ('Just featureSize ~ Eval (Last shape)),
    All KnownNat '[Eval (Length shape), featureSize, dim],
    All KnownNat shape,
    KnownDevice device,
    DivisionProofs featureSize 4,
    dim ~ Eval (QuaternionDimToNarrow (Eval (Length shape)))
  )

data ApplyToLast :: (a -> Exp b) -> [a] -> Exp [a]

data QuaternionComponent :: Nat -> Exp Nat

data Quaternion :: Nat -> Exp Nat

type instance Eval (QuaternionComponent n) = Div n 4

type instance Eval (Quaternion n) = n * 4

type instance Eval (ApplyToLast f '[]) = '[]

type instance Eval (ApplyToLast f (x ': '[])) = Eval (f x) ': '[]

type instance Eval (ApplyToLast f (x ': (y ': xs))) = x ': (Eval (ApplyToLast f (y ': xs)))

data FromJust :: Maybe a -> Exp a

type instance Eval (FromJust ('Just a)) = a

-- TODO: Support more dimensions than 2 and 3.
type QuaternionDimToNarrow =
  Case
    [ 2 --> 1,
      3 --> 2
    ]

-- | Accessors
r,
  i,
  j,
  k ::
    forall featureSize dim shape device dtype.
    ( HasQuaternionComponents shape dim featureSize device dtype
    ) =>
    Tensor device dtype shape ->
    Tensor device dtype (Eval (ApplyToLast QuaternionComponent shape))
r = narrow @dim @0 @(Div featureSize 4)
i = narrow @dim @(featureSize `Div` 4) @(featureSize `Div` 4)
j = narrow @dim @(featureSize `Div` 2) @(featureSize `Div` 4)
k = narrow @dim @(featureSize - (featureSize `Div` 4)) @(featureSize `Div` 4)

-- data Modulous = ModulousScalar | ModulousVector
--   deriving stock (Show, Eq, Ord, Enum, Bounded, Generic)

-- data SingModulous m where
--   SModulousScalar :: SingModulous 'ModulousScalar
--   SModulousVector :: SingModulous 'ModulousVector

-- type family Modulate (m :: Modulous) (f :: [Nat]) :: [Nat] where
--   Modulate 'ModulousScalar (b ': f ': '[]) = '[b]
--   Modulate 'ModulousVector '[b, f] = '[b, NQuaternions f]

-- type Modulateable dtype device shape =
--   ( SumDType dtype ~ dtype,

--     DivisionProofs (Eval(FromJust =<< (Last shape))) 4,
--     SumDTypeIsValid device dtype,
--     KnownDevice device,
--     StandardFloatingPointDTypeValidation
--       device
--       dtype,
--     All KnownNat shape
--   )

-- modulous ::
--   (Modulateable dtype device shape) =>
--   SingModulous s ->
--   Tensor device dtype shape ->
--   Tensor device dtype (Modulate s shape)
-- modulous sm t' = case sm of
--   SModulousScalar -> sumDim @1 root
--   SModulousVector -> root
--   where
--     root = sqrt $ (r' * r' + i' * i' + j' * j' + k' * k')
--     r' = r t'
--     i' = i t'
--     j' = j t'
--     k' = k t'

-- -- TODO: orig uses "repeate" instead of expand and has an extra expand_as step . . .
-- --
-- normalize ::
--   forall dtype device shape.
--   (Modulateable dtype device shape) =>
--   Tensor device dtype shape ->
--   Tensor device dtype shape
-- normalize t' = t' / (addScalar eps modu)
--   where
--     eps = 0.0001 :: Float
--     modu =
--       expand @shape True
--         . reshape @'[batchSize, 1]
--         $ modulous SModulousScalar t'

-- | Applies the Hamilton product of q0 and q1:
--    Shape:
--        - q0, q1 should be of some shape where the final dimension 
--        - are quaternions
--        (rr' - xx' - yy' - zz')  +
--        (rx' + xr' + yz' - zy')i +
--        (ry' - xz' + yr' + zx')j +
--        (rz' + xy' - yx' + zr')k +
hamilton ::
  forall shape featureSize dim dtype device lastDim.
  ( lastDim ~ Eval (Length shape),
    ('Just featureSize ~ Eval (Last shape)),
    All KnownNat '[lastDim, featureSize, dim],
    HasQuaternionComponents shape dim featureSize device dtype,
    dim ~ Eval (QuaternionDimToNarrow (Eval (Length shape)))
  ) =>
  Tensor device dtype shape ->
  Tensor device dtype shape ->
  Tensor device dtype shape
hamilton q0 q1 = cat @lastDim (r' :. i' :. j' :. k' :. HNil)
  where
    q1_r = r q1
    q1_i = i q1
    q1_j = j q1
    q1_k = k q1
    -- rr', xx', yy', and zz'
    r_base = q0 * q1
    -- (rr' - xx' - yy' - zz')
    r' = r r_base - i r_base - j r_base - k r_base
    -- rx', xr', yz', and zy'
    i_base = q0 * cat @lastDim ((q1_i :. q1_r :. q1_k :. q1_j :. HNil))
    -- (rx' + xr' + yz' - zy')
    i' = r i_base + i i_base + j i_base - k i_base
    -- ry', xz', yr', and zx'
    j_base = q0 * cat @lastDim ((q1_j :. q1_k :. q1_r :. q1_i :. HNil))
    --- (rx' + xr' + yz' - zy')
    j' = r j_base - i j_base + j j_base + k j_base
    -- rz', xy', yx', and zr'
    k_base = q0 * cat @lastDim (q1_k :. q1_j :. q1_i :. q1_r :. HNil)
    -- (rx' + xr' + yz' - zy')
    k' = r k_base + i k_base - j k_base + k k_base

infix 5 ⦿

(⦿) ::
  forall shape featureSize dim dtype device lastDim.
  ( lastDim ~ Eval (Length shape),
    ('Just featureSize ~ Eval (Last shape)),
    All KnownNat '[lastDim, featureSize, dim],
    HasQuaternionComponents shape dim featureSize device dtype,
    dim ~ Eval (QuaternionDimToNarrow (Eval (Length shape)))
  ) =>
  Tensor device dtype shape ->
  Tensor device dtype shape ->
  Tensor device dtype shape
a ⦿ b = hamilton a b

data InitializationScheme = InitializationSchemeGlorot | InitializationSchemeHe
  deriving stock (Show, Eq, Ord, Enum, Bounded, Generic)

data Quaternions shape device dtype = Quaternions
  { quaternions_r :: Tensor device dtype shape,
    quaternions_i :: Tensor device dtype shape,
    quaternions_j :: Tensor device dtype shape,
    quaternions_k :: Tensor device dtype shape
  }

catQuaternions ::
  forall shape lastDim outShape dtype device.
  ( KnownNat lastDim,
    lastDim ~ Eval (Length shape),
    '(outShape, dtype, device)
      ~ Cat
          (Eval (Length shape))
          '[ Tensor device dtype shape,
             Tensor device dtype shape,
             Tensor device dtype shape,
             Tensor device dtype shape
           ]
  ) =>
  Quaternions shape device dtype ->
  Tensor device dtype outShape
catQuaternions (Quaternions q_r q_i q_j q_k) = cat @lastDim (q_r :. q_i :. q_j :. q_k :. HNil)

-- | Weight initialization
initialize ::
  forall nIn nOut dtype device.
  ( SumDType dtype ~ dtype,
    KnownDevice device,
    StandardFloatingPointDTypeValidation
      device
      dtype,
    SumDTypeIsValid device dtype,
    All KnownNat '[nIn, nOut]
  ) =>
  IO (Quaternions '[nIn, nOut] device dtype)
initialize = do
  (modulus :: Tensor device dtype '[nIn, nOut]) <- UnsafeMkTensor <$> kaimingUniform FanIn (LeakyRelu $ P.sqrt (0.0 :: Float)) [fan_out, fan_in]
  (v :: Tensor device dtype '[nIn * nOut, 3]) <- UnsafeMkTensor <$> randUnif [fan_out * fan_in, 3]
  (phase :: Tensor device dtype '[nIn, nOut]) <- UnsafeMkTensor . (* pi) <$> randUnif [fan_out, fan_in]
  let vnorm = sqrt . sumDim @1 $ (addScalar eps (v ^ (2 :: Int)))
      v' = v / (expand @'[nIn * nOut, 3] True $ reshape @'[nIn * nOut, 1] vnorm)
      v_i = reshape @'[nIn, nOut] $ narrow @1 @0 @1 v'
      v_j = reshape @'[nIn, nOut] $ narrow @1 @1 @1 v'
      v_k = reshape @'[nIn, nOut] $ narrow @1 @2 @1 v'
      weight_r = modulus * cos phase
      weight_i = modulus * v_i * sin (phase)
      weight_j = modulus * v_j * sin (phase)
      weight_k = modulus * v_k * sin (phase)
  return $ Quaternions weight_r weight_i weight_j weight_k
  where
    pi = 3.141592653589793238462643383279502884197169399375105820974944
    eps = 0.0001 :: Float
    randUnif = fmap ((\x -> x - 1) . (* 2)) . randIO'
    fan_in = natValI @nIn
    fan_out = natValI @nOut
