{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.Bifunctor (Bifunctor (second), bimap)
import Data.Int (Int64)
import Data.List (transpose)
import Data.Maybe (fromMaybe)
import Debug.Trace
import GHC.TypeLits (KnownNat)
import Models.Word2Vec
  ( TrainableWord2Vec (TrainableWord2Vec),
    Word2Vec,
    Word2VecSpec (Word2VecSpec),
    trainAndMonitor,
  )
import Path (Abs, File, absfile)
import Streamly
  ( AheadT,
    MonadAsync,
    SerialT,
    aheadly,
  )
import qualified Streamly.Prelude as S
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Streamly.Dataloader
  ( NegativeSample (unNegativeSample),
    SymbolOrToken (unSymbolOrToken),
    TokenStreamDataset (..),
    dataset,
    debugPrintTokenStream,
    inBatchesOf,
    trainStream,
  )
import Torch.Tensor (asTensor)
import Torch.Typed
  ( DeviceType (CPU),
    KnownDevice,
    Parameterized (flattenParameters),
    Randomizable (sample),
    Tensor (..),
    natValI,
    zeros,
  )
import Torch.Typed.Optim
import Data.Int (Int32)
type EmebddingDimension = 256

type BatchSize = 128

type WindowSize = 5

main :: IO ()
main = do
  initModel <- model
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel) -- GD
  train <- trainSet
  print $ Prelude.length $ tokenStreamDatasetToken train
  print $ tokenStreamDatasetOccurances train
  valid <- validSet
  trainAndMonitor @BatchSize @WindowSize
    "mon.html"
    (TrainableWord2Vec initModel)
    initOptim
    (niceData2FoulTensorDataSet @'(D.CPU, 0) @BatchSize . inBatchesOf bs $ trainStream ws train)
    (niceData2FoulTensorDataSet @'(D.CPU, 0) @BatchSize . inBatchesOf bs $ trainStream ws valid)
  where
    bs = natValI @BatchSize
    ws = natValI @WindowSize

debugPrintTokenStream' :: (MonadAsync m) => TokenStreamDataset SymbolOrToken -> AheadT m [Int32] -> AheadT m [Int32]
debugPrintTokenStream' = debugPrintTokenStream (fromMaybe (Left ' ') . fmap unSymbolOrToken)

model :: IO (Word2Vec WindowSize 27335 EmebddingDimension 'D.Float '( 'CPU, 0))
model = sample $ Word2VecSpec @WindowSize @27335 @EmebddingDimension @'D.Float @'( 'D.CPU, 0)

trainSet :: IO (TokenStreamDataset SymbolOrToken)
trainSet = dataset [absfile|/Users/timpierson/arity/word2nion/data/wikitext-2/train.txt|]

validSet :: IO (TokenStreamDataset SymbolOrToken)
validSet = dataset [absfile|/Users/timpierson/arity/word2nion/data/wikitext-2/valid.txt|]

niceData2FoulTensorDataSet ::
  forall device batchSize m.
  (KnownNat batchSize, KnownDevice device, MonadAsync m) =>
  AheadT m [([Int32], NegativeSample)] ->
  SerialT m (([Tensor device 'D.Int64 '[batchSize]], [Tensor device 'D.Int64 '[batchSize]]), Tensor device 'D.Float '[batchSize])
niceData2FoulTensorDataSet =
  aheadly . S.map ((,zeros) . bimap alaTensor alaTensor . unzip . fmap (second unNegativeSample))
  where
    alaTensor =
      fmap (UnsafeMkTensor . asTensor . fmap (fromIntegral :: Int32 -> Int64))
        . Data.List.transpose
