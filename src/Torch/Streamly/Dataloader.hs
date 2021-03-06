{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -O2 #-}

-- | Module defines types and functions for dataset streams
-- to be fead to models during training.  `trainStream` provides continuous stream of tokens and a stream of negatively sampled tokens
-- from a given TokenStreamDataset.
module Torch.Streamly.Dataloader
  ( TokenStreamDataset (..),
    SymbolOrToken (..),
    NegativeSample(..),
    dataset,
    trainStream,
    byTwos,
    inBatchesOf,
    indexStreamToTokenStream,
    debugPrintTokenStream
  )
where

import Control.Monad.Catch (MonadCatch)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.Bifunctor (bimap)
import qualified Data.ByteString.Builder as Builder
import qualified Data.ByteString.Internal as BS (c2w)
import qualified Data.ByteString.Lazy as BSL
import Data.IntervalMap.Generic.Strict (IntervalMap)
import qualified Data.IntervalMap.Generic.Strict as IM
import qualified Data.IntervalMap.Generic.Strict as IvMap
import Data.List (sortOn)
import Data.Map.Monoidal.Strict (MonoidalMap)
import qualified Data.Map.Monoidal.Strict as MMap
import qualified Data.Maybe as Maybe (mapMaybe)
import Data.Monoid (First (..), Sum (..))
import qualified Data.Text as T
import Data.Text.Encoding.Error (lenientDecode)
import qualified Data.Text.Encoding as T
import Data.Word (Word8)
import Foreign.Storable (Storable)
import GHC.Generics (Generic)
import Path (File, Path, toFilePath)
import Streamly (AheadT, MonadAsync, aheadly, serially)
import qualified Streamly.Data.Fold as FL
import qualified Streamly.Internal.Data.Fold as FL
import qualified Streamly.Internal.FileSystem.File as File
import qualified Streamly.Internal.Memory.Array as SA
import qualified Streamly.Prelude as S
import Data.Int (Int32)
import System.Random (randomRIO)
import Torch.Streamly.Tokenizer (Token (Symbol, Token), tokenize)

-- | Represents a dataset of some token type `a`
data TokenStreamDataset a = TokenStreamDataset
  { -- | The lookup map of tokens to integers
    tokenStreamDatasetToken :: MonoidalMap a (First Int32),
    -- | The buffered, tokenized corpora in a streamly Array for O(1) random access
    tokenStreamDatasetData :: SA.Array Int32,
    -- | The counts of the tokens in the corpora
    tokenStreamDatasetFrequencies :: MonoidalMap a (Sum Int32),
    -- | The probability intervals per token (for sampling from the noise distribution)
    tokenStreamDatasetProbabilities :: IntervalMap (StrictTuple Double Double) Int32,
    -- | The total number of examples
    tokenStreamDatasetOccurances :: Int32
  }
  deriving stock (Eq, Show, Ord, Generic)

-- These are not so efficient, but it was convienient to be able to merge to two datasets so easily.
--
instance (Ord a) => Semigroup (TokenStreamDataset a) where
  (<>) (TokenStreamDataset _ da fa _ oa) (TokenStreamDataset _ db fb _ ob) = TokenStreamDataset indexLookup (da <> db) f' p' o
    where
      indexLookup = makeLookup' f'
      f' = fa <> fb
      o = oa + ob
      mm = MMap.mapKeys (fromJust . getFirst . (indexLookup MMap.!)) f'
      p' = toIvMap o mm
      fromJust (Just v) = v
      fromJust _ = error "Semigroup (TokenStreamDataset a): it is impossible that fromJust is partial here!  Something is wrong!"

makeLookup' :: (Ord k) => MonoidalMap k a -> MonoidalMap k (First Int32)
makeLookup' mm' = MMap.fromList $ zip (MMap.keys mm') (First . Just <$> [0 ..])

instance (Ord a) => Monoid (TokenStreamDataset a) where
  mempty = TokenStreamDataset mempty (SA.fromList []) mempty (IM.fromList []) 0
  mappend = (<>)

data StrictTuple a b = StrictTuple !a !b
  deriving stock (Eq, Show, Ord, Generic)

instance Ord e => IM.Interval (StrictTuple e e) e where
  lowerBound (StrictTuple a _) = a
  upperBound (StrictTuple _ b) = b
  rightClosed _ = False

-- | Use our tokenizer on a file. Expects text not structured data.
tokenStream :: (MonadAsync m, MonadCatch m) => Path a File -> AheadT m Token
tokenStream fp =
  S.concatMap (S.fromList . handleIt . tokenize . BSL.toStrict . Builder.toLazyByteString . foldMap Builder.word8)
   . S.splitOnSuffix splitChars FL.toList
    $ File.toBytes (toFilePath fp)
  where
    handleIt (Left err) = error (show err)
    handleIt (Right bs) = bs

{-# INLINE splitChars #-}
splitChars :: Word8 -> Bool
splitChars s
  | s == periodMark = True
  | s == questionMark = True
  | s == exclaimationMark = True
  | otherwise = False

{-# INLINE questionMark #-}
questionMark :: Word8
questionMark = m
  where
    !m = BS.c2w '?'

{-# INLINE periodMark #-}
periodMark :: Word8
periodMark = m
  where
    !m = BS.c2w '.'

{-# INLINE exclaimationMark #-}
exclaimationMark :: Word8
exclaimationMark = m
  where
    !m = BS.c2w '!'

newtype SymbolOrToken = SymbolOrToken {unSymbolOrToken :: Either Char T.Text}
  deriving stock (Eq, Ord, Generic, Show)

filterAndLowercase :: (MonadAsync m) => AheadT m Token -> AheadT m (Either Char T.Text)
filterAndLowercase = S.mapMaybe go
  where
    go (Token t) = Just . Right $ T.toLower . T.decodeUtf8With lenientDecode $ t
    go (Symbol t) = Just . Left $ t
    go _ = Nothing

-- | Given a function for maybe extracting keys and a function
-- for extracting values, produce a `Fold` to a `MonoidalMap` of keys and values
{-# INLINE groupBy #-}
groupBy ::
  (Monad m, Ord k, Monoid b) =>
  (a -> Maybe k) ->
  (a -> b) ->
  FL.Fold m a (MonoidalMap k b)
groupBy kf vf = FL.Fold step' initial' extract'
  where
    initial' = return mempty
    step' kv a = return $ maybe kv (\k -> kv <> MMap.singleton k (vf a)) (kf a)
    extract' = return

tokenFrequencies :: (MonadAsync m, Ord a) => AheadT m a -> m (MonoidalMap a (Sum Int32))
tokenFrequencies = S.fold (groupBy Just (const $ Sum 1)) . serially . aheadly

{-# INLINE byTwos #-}

-- | Two by two . . .
--
-- >>> byTwos [1..10]
-- [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]
byTwos :: [b] -> [(b, b)]
byTwos xs = zip xs $ tail xs

toIvMap :: Int32 -> MonoidalMap a (Sum Int32) -> IntervalMap (StrictTuple Double Double) a
toIvMap total mm =
  IvMap.fromList . Maybe.mapMaybe go . byTwos $ xs
  where
    xs = bimap Just ((/ fromIntegral total) . fromIntegral . getSum) <$> sortOn snd (MMap.toAscList mm)
    go ((Just a, lb), (_, ub)) = Just (StrictTuple lb ub, a)
    go _ = Nothing

-- | Make a dataset from a file.  This implementaiton is a bit annoying because whatever we put in the
-- Array needs to have a Storeable instance.  This means that we can't do unbounded containers
-- and rolling our own datatypes requres writing the storeable instance.  So we move everything to Int maps
-- and store the array of Ints.
dataset :: (MonadAsync m) => Path a File -> m (TokenStreamDataset SymbolOrToken)
dataset fp = do
  prepped <- liftIO . S.toList . serially . aheadly . filterAndLowercase $ tokenStream fp
  mm' <- tokenFrequencies $ S.fromList prepped
  let total = getSum . mconcat $ MMap.elems mm'
  let indexLookup = MMap.fromList $ zip (MMap.keys mm') (First . Just <$> [0 ..])
  let mm = MMap.mapKeys (fromJust . getFirst . (indexLookup MMap.!)) mm'
  prepped' <- SA.fromStream . serially . aheadly . S.map (fromJust . getFirst . (indexLookup MMap.!)) $ S.fromList prepped
  pure $ TokenStreamDataset (MMap.mapKeys SymbolOrToken indexLookup) prepped' (MMap.mapKeys SymbolOrToken mm') (toIvMap total mm) total
  where
    fromJust (Just v) = v
    fromJust _ = error "dataset: it is impossible that fromJust is partial here!  Something is wrong!"

-- | Subsample infrequent words occurding to a probability derived from the words frequency
-- in the corpora. The original word2vec paper used the same caclulation as below on
-- the input word itself.  Since we're considering lists of words, we use the minimum value
-- of all the words in the list since we want to increase the chance that rare words are retained.
subsampleFilter :: (MonadIO m, Ord a, Show a) => Int32 -> MonoidalMap a (Sum Int32) -> [a] -> m (Maybe [a])
subsampleFilter total mm d = do
  (p :: Double) <- liftIO $ randomRIO (0, 1)
  pure $ case minimum (calc mm <$> d) > p of
    True -> Nothing -- Higer values of calc should be higher values of discarding, so if p is less than that value, discard.
    False -> Just d
  where
    calc mm' v = 1 - sqrt (t / f_wi)
      where
        t = 10e-5 -- threashold from the paper
        lkp = case MMap.lookup v mm' of 
              Just v' -> v' 
              Nothing -> Sum 0 -- error ("Failsauce: " <> show d)
        f_wi = fromIntegral (getSum lkp) / fromIntegral total -- Frequency of word i

-- | do these in batches
negativeSample :: (MonadAsync m) => IntervalMap (StrictTuple Double Double) a -> AheadT m a
negativeSample im = S.concatMapM return $ do
  (p :: Double) <- liftIO $ randomRIO (0, 1)
  let c = IvMap.containing im p 
  case IvMap.null c of 
    True -> pure . S.yield . snd . head $ IvMap.toDescList im
    False -> pure . S.yield . snd . IvMap.findLast $ c

negativeSampleStream :: (MonadAsync m) => IntervalMap (StrictTuple Double Double) a -> AheadT m a
negativeSampleStream im = negativeSample im <> negativeSampleStream im

-- | Infinite stream of contiguous chunks of size `s`
shufflingStreamOfSize :: (MonadAsync m, Storable a) => Int32 -> SA.Array a -> AheadT m [a]
shufflingStreamOfSize s = S.chunksOf (fromIntegral s) FL.toList . shufflingStreamOfSize' s

-- | Infinite stream of samples from an array of some size
shufflingStreamOfSize' :: (MonadAsync m, Storable a) => Int32 -> SA.Array a -> AheadT m a
shufflingStreamOfSize' s a = sample s a <> shufflingStreamOfSize' s a

-- | Randomly samples a slice of size `size` from the array, returning it as a stream.
sample :: (MonadAsync m, Storable a) => Int32 -> SA.Array a -> AheadT m a
sample size a = S.concatMapM return $ do
  start <-  liftIO $ fromIntegral <$> randomRIO (0,  a' - size)
  pure . S.fromList $ fmap (SA.unsafeIndex a) [start .. (start + fromIntegral size)]
  where 
    a' = fromIntegral (SA.length a)

newtype NegativeSample = NegativeSample {unNegativeSample :: [Int32]}
 deriving stock (Eq, Show, Ord, Generic)

-- | Make a training stream of token indicies of size windowSize and a negative
-- sampling stream of size windowSize 
trainStream :: (MonadAsync m, Ord a) => Int -> TokenStreamDataset a -> AheadT m ([Int32], NegativeSample)
trainStream windowSize (TokenStreamDataset indexLookup a mm im c) =  (,) <$> trainStream' <*> (S.map NegativeSample . S.chunksOf windowSize FL.toList  $ negativeSampleStream im)
  where
    trainStream' = S.mapMaybeM (subsampleFilter c (MMap.mapKeys (fromJust . getFirst . (indexLookup MMap.!)) mm)) 
       $ shufflingStreamOfSize (fromIntegral windowSize) a
    fromJust (Just v) = v
    fromJust _ = error "trainStream: it is impossible that fromJust is partial here!  Something is wrong!"


indexLookupToTokenLookup :: MonoidalMap a (First Int32) -> MonoidalMap Int32 (First a)
indexLookupToTokenLookup = MMap.fromList . map (\(k,v) -> (v, First k)) . Maybe.mapMaybe (sequence . bimap Just getFirst) . MMap.toList

inBatchesOf :: Monad m => Int -> AheadT m a -> AheadT m [a]
inBatchesOf s = S.chunksOf s FL.toList

indexToToken :: MonoidalMap Int32 (First a) -> Int32 -> Maybe a
indexToToken mm i = getFirst =<< MMap.lookup i mm

indiciesToTokens :: MonoidalMap Int32 (First a) -> [Int32] -> [Maybe a]
indiciesToTokens mm = fmap (indexToToken mm)

indexStreamToTokenStream' :: (MonadAsync m) => MonoidalMap Int32 (First a) -> AheadT m [Int32] -> AheadT m [Maybe a]
indexStreamToTokenStream' mm = S.map (indiciesToTokens mm)

indexStreamToTokenStream :: (MonadAsync m) => TokenStreamDataset a -> AheadT m [Int32] -> AheadT m [Maybe a]
indexStreamToTokenStream tsd = indexStreamToTokenStream' (indexLookupToTokenLookup (tokenStreamDatasetToken tsd ))

debugPrintTokenStream :: (Show b, MonadAsync m) => (Maybe a -> b) -> TokenStreamDataset a -> AheadT m [Int32] -> AheadT m [Int32]
debugPrintTokenStream f tsd = S.mapM go
    where 
      go x = do 
        liftIO $ print $ f <$> convert x
        pure x
      convert = indiciesToTokens (indexLookupToTokenLookup (tokenStreamDatasetToken tsd))