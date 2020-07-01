{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE UnicodeSyntax               #-}
{-# LANGUAGE DataKinds               #-}
{-# LANGUAGE DeriveGeneric           #-}
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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- | Module defining operations on quaternions.
-- Ported from https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py
--
module Torch.Typed.Quaternion where
import           GHC.TypeLits
import           GHC.TypeNats           ()
import           Prelude                hiding (sqrt, sum)
import qualified Torch.Internal.Class   as ATen
import           Torch.Typed
import qualified Torch.Tensor           as D


type family NQuaternions (features :: Nat) where
    NQuaternions n = Div n 4

type HasQuaternion f = ((f `Mod` 4) ~ 0)

type DivisionProofs n d = (n `Div` d <= n
  , Div n d + Div n d <= n
  , Div n (d `Div` 2) + Div n d <= n
  , (n - Div n d)+ Div n d <= n
  , (Div n d * d)
                    ~ ((n - Div n d) + Div n d) )

r, i, j, k :: forall batchSize featureSize device dtype. (DivisionProofs featureSize 4, All KnownNat '[batchSize, featureSize]) => Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, NQuaternions featureSize]
r = narrow @1 @0 @(Div featureSize 4)
i = narrow @1 @(featureSize `Div` 4) @(featureSize `Div` 4)
j = narrow @1 @(featureSize `Div` 2) @(featureSize `Div` 4)
k = narrow @1 @(featureSize - (featureSize `Div` 4)) @(featureSize `Div` 4)


data Modulous = ModulousScalar | ModulousVector

data SingModulous m where
    SModulousScalar :: SingModulous 'ModulousScalar
    SModulousVector :: SingModulous 'ModulousVector

type family Modulate (m :: Modulous) (f :: [Nat]) where
    Modulate 'ModulousScalar (b ': f ': '[]) = '[b]
    Modulate 'ModulousVector '[b,f] = '[b, NQuaternions f]

type Modulateable dtype device batchSize featureSize
  = (SumDType dtype ~ dtype,
    DivisionProofs featureSize 4,
     SumDTypeIsValid device dtype
     , KnownDevice device
     , StandardFloatingPointDTypeValidation
                      device dtype
                      , All KnownNat '[batchSize, featureSize])

modulous :: (Modulateable dtype device batchSize featureSize) => SingModulous s -> Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype (Modulate s '[batchSize, featureSize])
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
normalize :: forall dtype device batchSize featureSize. (Modulateable dtype device batchSize featureSize) => Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, featureSize]
normalize t' = t' / (addScalar eps modu)
    where
        eps = 0.0001 :: Float
        modu = expand @'[batchSize, featureSize] True . reshape @'[batchSize, 1] $ modulous SModulousScalar t'

qcat :: forall batchSize featureSize (dtype :: DType)
                    (device :: (DeviceType, Nat)) (tensors :: [k]) .
                    (KnownNat 1, '( '[batchSize, featureSize], dtype, device) ~ Cat 1 tensors,
                    ATen.Castable (HList tensors) [D.ATenTensor]) => HList tensors -> Tensor device dtype '[batchSize, featureSize]
qcat = cat @1

-- | Applies the Hamilton product of q0 and q1:
--    Shape:
--        - q0, q1 should be '[batch_size, quaternion_number]
--        (rr' - xx' - yy' - zz')  +
--        (rx' + xr' + yz' - zy')i +
--        (ry' - xz' + yr' + zx')j +
--        (rz' + xy' - yx' + zr')k +
--
hamilton :: forall dtype device batchSize featureSize. (Modulateable dtype device batchSize featureSize) => Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, featureSize]
hamilton q0 q1 = qcat ( r' :. i' :. j' :. k' :. HNil)
    where
        q1_r = r q1
        q1_i = i q1
        q1_j = j q1
        q1_k = k q1
        -- rr', xx', yy', and zz'
        r_base = q0 * q1
        -- (rr' - xx' - yy' - zz')
        r'  = r r_base - i r_base - j r_base - k r_base
        -- rx', xr', yz', and zy'
        i_base = q0 * (qcat ((q1_i :. q1_r :. q1_k :. q1_j :. HNil) ))
        -- (rx' + xr' + yz' - zy')
        i'   = r i_base + i i_base + j i_base - k i_base
        -- ry', xz', yr', and zx'
        j_base = q0 * cat @1 ((q1_j :. q1_k :. q1_r :. q1_i :. HNil) )
        --- (rx' + xr' + yz' - zy')
        j'   = r j_base - i j_base + j j_base + k j_base
        -- rz', xy', yx', and zr'
        k_base = q0 * cat @1 ( q1_k :. q1_j :. q1_i :. q1_r :. HNil)
        -- (rx' + xr' + yz' - zy')
        k'   = r k_base + i k_base - j k_base + k k_base

infix 5 ⦿

(⦿) :: forall dtype device batchSize featureSize. (Modulateable dtype device batchSize featureSize) => Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, featureSize] -> Tensor device dtype '[batchSize, featureSize]
a ⦿ b = hamilton a b