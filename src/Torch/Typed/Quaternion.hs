{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE DataKinds               #-}
{-# LANGUAGE DeriveGeneric           #-}
{-# LANGUAGE DerivingStrategies      #-}
{-# LANGUAGE FlexibleContexts        #-}
{-# LANGUAGE FlexibleInstances       #-}
{-# LANGUAGE GADTs                   #-}
{-# LANGUAGE MultiParamTypeClasses   #-}
{-# LANGUAGE NoImplicitPrelude       #-}
{-# LANGUAGE NoStarIsType            #-}
{-# LANGUAGE OverloadedLists         #-}
{-# LANGUAGE PartialTypeSignatures   #-}
{-# LANGUAGE PolyKinds               #-}
{-# LANGUAGE RankNTypes              #-}
{-# LANGUAGE ScopedTypeVariables     #-}
{-# LANGUAGE TypeApplications        #-}
{-# LANGUAGE TypeFamilies            #-}
{-# LANGUAGE TypeOperators           #-}
{-# LANGUAGE UndecidableInstances    #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax           #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

-- | Module defining operations on quaternions.
-- Ported from https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py
module Torch.Typed.Quaternion
  ( (⦿),
    hamilton,
    initialize,
    normalize,
    Quaternions (..),
    catQuaternions,
    HasQuaternion,
    NQuaternions,
    DivisionProofs,
  )
where

import           GHC.Generics          (Generic)
import           GHC.TypeLits
import           GHC.TypeNats          ()
import           Prelude               hiding (cos, pi, sin, sqrt, sum)
import qualified Prelude               as P (sqrt)
import           Torch.Initializers    (FanMode (..), NonLinearity (..),
                                        kaimingUniform)
import qualified Torch.Internal.Class  as ATen
import qualified Torch.Tensor          as D
import           Torch.TensorFactories (randIO')
import           Torch.Typed

type family NQuaternions (features :: Nat) where
  NQuaternions n = Div n 4

type HasQuaternion f = ((f `Mod` 4) ~ 0)

type DivisionProofs n d =
  ( n `Div` d <= n,
    Div n d + Div n d <= n,
    Div n (d `Div` 2) + Div n d <= n,
    (n - Div n d) + Div n d <= n,
    (Div n d * d)
      ~ ((n - Div n d) + Div n d)
  )

-- | Accessors
r,
  i,
  j,
  k ::
    forall batchSize featureSize device dtype.
    (DivisionProofs featureSize 4, All KnownNat '[batchSize, featureSize]) =>
    Tensor device dtype '[batchSize, featureSize] ->
    Tensor device dtype '[batchSize, NQuaternions featureSize]
r = narrow @1 @0 @(Div featureSize 4)
i = narrow @1 @(featureSize `Div` 4) @(featureSize `Div` 4)
j = narrow @1 @(featureSize `Div` 2) @(featureSize `Div` 4)
k = narrow @1 @(featureSize - (featureSize `Div` 4)) @(featureSize `Div` 4)

data Modulous = ModulousScalar | ModulousVector
  deriving stock (Show, Eq, Ord, Enum, Bounded, Generic)

data SingModulous m where
  SModulousScalar :: SingModulous 'ModulousScalar
  SModulousVector :: SingModulous 'ModulousVector

type family Modulate (m :: Modulous) (f :: [Nat]) :: [Nat] where
  Modulate 'ModulousScalar (b ': f ': '[]) = '[b]
  Modulate 'ModulousVector '[b, f] = '[b, NQuaternions f]

type Modulateable dtype device batchSize featureSize =
  ( SumDType dtype ~ dtype,
    DivisionProofs featureSize 4,
    SumDTypeIsValid device dtype,
    KnownDevice device,
    StandardFloatingPointDTypeValidation
      device
      dtype,
    All KnownNat '[batchSize, featureSize]
  )

modulous ::
  (Modulateable dtype device batchSize featureSize) =>
  SingModulous s ->
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype (Modulate s '[batchSize, featureSize])
modulous sm t' = case sm of
  SModulousScalar -> sumDim @1 root
  SModulousVector -> root
  where
    root = sqrt $ (r' * r' + i' * i' + j' * j' + k' * k')
    r' = r t'
    i' = i t'
    j' = j t'
    k' = k t'

-- TODO: orig uses "repeate" instead of expand and has an extra expand_as step . . .
--
normalize ::
  forall dtype device batchSize featureSize.
  (Modulateable dtype device batchSize featureSize) =>
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype '[batchSize, featureSize]
normalize t' = t' / (addScalar eps modu)
  where
    eps = 0.0001 :: Float
    modu =
      expand @'[batchSize, featureSize] True
        . reshape @'[batchSize, 1]
        $ modulous SModulousScalar t'

-- | Helper that specializes the cat function for the shapes
-- needed below.  Helps type inference work out that `hamilton`
-- is well typed.
qcat ::
  forall
    batchSize
    featureSize
    (dtype :: DType)
    (device :: (DeviceType, Nat))
    (tensors :: [k]).
  ( KnownNat 1,
    '( '[batchSize, featureSize], dtype, device) ~ Cat 1 tensors,
    ATen.Castable (HList tensors) [D.ATenTensor]
  ) =>
  HList tensors ->
  Tensor device dtype '[batchSize, featureSize]
qcat = cat @1

-- | Applies the Hamilton product of q0 and q1:
--    Shape:
--        - q0, q1 should be '[batch_size, quaternion_number]
--        (rr' - xx' - yy' - zz')  +
--        (rx' + xr' + yz' - zy')i +
--        (ry' - xz' + yr' + zx')j +
--        (rz' + xy' - yx' + zr')k +
hamilton ::
  forall dtype device batchSize featureSize.
  (Modulateable dtype device batchSize featureSize) =>
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype '[batchSize, featureSize]
hamilton q0 q1 = qcat (r' :. i' :. j' :. k' :. HNil)
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
    i_base = q0 * (qcat ((q1_i :. q1_r :. q1_k :. q1_j :. HNil)))
    -- (rx' + xr' + yz' - zy')
    i' = r i_base + i i_base + j i_base - k i_base
    -- ry', xz', yr', and zx'
    j_base = q0 * cat @1 ((q1_j :. q1_k :. q1_r :. q1_i :. HNil))
    --- (rx' + xr' + yz' - zy')
    j' = r j_base - i j_base + j j_base + k j_base
    -- rz', xy', yx', and zr'
    k_base = q0 * cat @1 (q1_k :. q1_j :. q1_i :. q1_r :. HNil)
    -- (rx' + xr' + yz' - zy')
    k' = r k_base + i k_base - j k_base + k k_base

infix 5 ⦿

(⦿) ::
  forall dtype device batchSize featureSize.
  (Modulateable dtype device batchSize featureSize) =>
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype '[batchSize, featureSize] ->
  Tensor device dtype '[batchSize, featureSize]
a ⦿ b = hamilton a b

data InitializationScheme = InitializationSchemeGlorot | InitializationSchemeHe
  deriving stock (Show, Eq, Ord, Enum, Bounded, Generic)

data Quaternions batch features device dtype = Quaternions
  { quaternions_r :: Tensor device dtype '[batch, features],
    quaternions_i :: Tensor device dtype '[batch, features],
    quaternions_j :: Tensor device dtype '[batch, features],
    quaternions_k :: Tensor device dtype '[batch, features]
  }

catQuaternions :: Quaternions batch features device dtype -> Tensor device dtype '[batch, features * 4]
catQuaternions (Quaternions q_r q_i q_j q_k) = cat @1 (q_r :. q_i :. q_j :. q_k :. HNil)

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
  IO (Quaternions nIn nOut device dtype)
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
