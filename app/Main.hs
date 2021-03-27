{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.Int (Int64)
import Data.List (transpose)
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
    SerialT,
    aheadly, MonadAsync
  )
import qualified Streamly.Prelude as S
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Streamly.Dataloader
  ( SymbolOrToken,
    TokenStreamDataset (..),
    dataset,
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
    zeros,
  )
import Torch.Typed.Optim
import Debug.Trace
main :: IO ()
main = do
  initModel <- model
  let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
  train <- trainSet
  print $ Prelude.length $ tokenStreamDatasetToken train
  print $ tokenStreamDatasetOccurances train
  valid <- validSet
  trainAndMonitor @1 @5
    "mon.html"
    (TrainableWord2Vec initModel)
    initOptim
    (niceData2FoulTensorDataSet @'(D.CPU, 0) @1 . inBatchesOf 1 . fst $ trainStream 5 train)
    (niceData2FoulTensorDataSet @'(D.CPU, 0) @1 . inBatchesOf 1 . fst $ trainStream 5 valid)

model :: IO (Word2Vec 5 27335 128 'D.Float '( 'CPU, 0))
model = sample $ Word2VecSpec @5 @27335 @128 @'D.Float @'( 'D.CPU, 0)

trainSet :: IO (TokenStreamDataset SymbolOrToken)
trainSet = dataset [absfile|/Users/timpierson/arity/word2nion/data/wikitext-2/train.txt|]

validSet :: IO (TokenStreamDataset SymbolOrToken)
validSet = dataset [absfile|/Users/timpierson/arity/word2nion/data/wikitext-2/valid.txt|]

niceData2FoulTensorDataSet ::
  forall device batchSize m.
  (KnownNat batchSize, KnownDevice device, MonadAsync m) =>
  AheadT m [[Int]] ->
  SerialT m ([Tensor device 'D.Int64 '[batchSize]], Tensor device 'D.Float '[batchSize])
niceData2FoulTensorDataSet =
  aheadly 
    . S.mapM (go .
      ( (, zeros)
          . fmap (UnsafeMkTensor . asTensor . fmap (fromIntegral :: Int -> Int64))
          . Data.List.transpose
      ))
  where
    go :: MonadIO m => ([Tensor device 'D.Int64 '[batchSize]], Tensor device 'D.Float '[batchSize]) -> m ([Tensor device 'D.Int64 '[batchSize]], Tensor device 'D.Float '[batchSize])
    go x = do
      liftIO $ print $ length $ fst x
      pure x