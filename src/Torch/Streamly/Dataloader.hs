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

module Torch.Streamly.Dataloader
  ( TokenStreamDataset (..),
    trainStream,
  )
where

import Control.Monad.Catch (MonadCatch)
import Control.Monad.IO.Class
  ( MonadIO,
    liftIO,
  )
import Data.Bifunctor (bimap)
import qualified Data.IntervalMap.Generic.Strict as IM
import Data.IntervalMap.Generic.Strict (IntervalMap)
import qualified Data.IntervalMap.Generic.Strict as IvMap
import Data.List (sortOn)
import qualified Data.Map.Monoidal.Strict as MMap
import Data.Map.Monoidal.Strict (MonoidalMap)
import qualified Data.Maybe as Maybe (mapMaybe)
import Data.Monoid (Sum (..))
import qualified Data.Set as Set
import qualified Data.Text as T
import Data.Text (Text)
import Foreign.Storable ()
import Foreign.Storable.Generic
import GHC.Generics (Generic)
import Path
import Streamly
import qualified Streamly.Data.Fold as FL
import qualified Streamly.Internal.Data.Fold as FL
import Streamly.Internal.Data.Unicode.Stream
  ( decodeUtf8,
  )
import qualified Streamly.Internal.FileSystem.File as File
import qualified Streamly.Internal.Memory.Array as SA
import Streamly.Prelude as S hiding (minimum, tail)
import System.Random (randomRIO)
import Torch.Streamly.Tokenizer

-- | Represents a dataset of some token type `a`
data TokenStreamDataset a = TokenStreamDataset
  { -- | The buffered, tokenized corpora in a streamly Array for O(1) random access
    tokenStreamDatasetData :: SA.Array a,
    -- | The frequencies of the tokens in the corpora
    tokenStreamDatasetFrequencies :: MonoidalMap a (Sum Int),
    -- | The probability intervals per token (for sampling from the noise distribution)
    tokenStreamDatasetProbabilities :: IntervalMap (StrictTuple Double Double) a,
    -- | The total number of examples
    tokenStreamDatasetOccurances :: Int
  }
  deriving stock (Eq, Show, Ord, Generic)

data StrictTuple a b = StrictTuple !a !b
  deriving stock (Eq, Show, Ord, Generic)

instance Ord e => IM.Interval (StrictTuple e e) e where
  lowerBound (StrictTuple a _) = a
  upperBound (StrictTuple _ b) = b
  rightClosed _ = False

tokenStream :: (MonadAsync m, MonadCatch m) => Path a File -> AheadT m Token
tokenStream fp =
  S.concatMap (S.fromList . tokenize . T.pack)
    . S.splitOnSuffix (`Set.member` splitChars) FL.toList
    . decodeUtf8
    $ File.toBytes (toFilePath fp)
  where
    splitChars = Set.fromList $ ['.', '!', '?']

newtype SymbolOrToken = SymbolOrToken {unSymbolOrToken :: Either Char Text}
  deriving stock (Eq, Ord, Generic)
  deriving newtype (GStorable)

deriving instance (GStorable (Either Char Text))

filterAndLowercase :: (MonadAsync m) => AheadT m Token -> AheadT m SymbolOrToken
filterAndLowercase = S.mapMaybe go
  where
    go (Token t) = Just . SymbolOrToken . Right $ T.toLower t
    go (Symbol t) = Just . SymbolOrToken . Left $ t
    go (Symbol t) = Just . SymbolOrToken . Left $ t
    go _ = Nothing

{-# INLINE groupBy #-}

-- | Given a function for maybe extracting keys and a function
-- for extracting values, produce a `Fold` to a `MonoidalMap` of keys and values
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

tokenFrequencies :: (MonadAsync m) => AheadT m SymbolOrToken -> m (MonoidalMap SymbolOrToken (Sum Int))
tokenFrequencies = S.fold (groupBy Just (const $ Sum 1)) . serially . aheadly

byTwos :: [b] -> [(b, b)]
byTwos xs = zip xs $ tail xs

toIvMap :: Int -> MonoidalMap a (Sum Int) -> IntervalMap (StrictTuple Double Double) a
toIvMap total mm =
  IvMap.fromList . Maybe.mapMaybe go . byTwos $ xs
  where
    xs = bimap Just ((/ fromIntegral total) . fromIntegral . getSum) <$> sortOn snd (MMap.toAscList mm)
    go ((Just a, lb), (_, ub)) = Just $ ((StrictTuple lb ub), a)
    go _ = Nothing

dataset :: (MonadAsync m) => Path a File -> m (TokenStreamDataset SymbolOrToken)
dataset fp = do
  prepped <- liftIO . SA.fromStream . serially . aheadly . filterAndLowercase $ tokenStream fp
  let stream = SA.toStream prepped
  mm <- tokenFrequencies $ stream
  let total = getSum . mconcat $ MMap.elems mm
  pure $ TokenStreamDataset prepped mm (toIvMap total mm) total

-- | Subsample infrequent words occurding to a probability derived from the words frequency
-- in the corpora. The original paper used the same caclulation as below on
-- the input word itself.  Since we're considering lists of words, we use the minimum value
-- of all the words in the list since we want to increase the chance that rare words are retained.
subsampleFilter :: (MonadIO m, Ord a) => Int -> MonoidalMap a (Sum Int) -> [a] -> m (Maybe [a])
subsampleFilter total mm d = do
  (p :: Double) <- liftIO $ randomRIO (0, 1)
  pure $ case (minimum $ calc mm <$> d) > p of
    True -> Nothing -- Higer values of calc should be higher values of discarding, so if p is less than that value, discard.
    False -> Just d
  where
    calc mm' v = 1 - (sqrt (t / f_wi))
      where
        t = 10e-5 -- threashold from the paper
        f_wi = fromIntegral (getSum (mm' MMap.! v)) / fromIntegral total -- Frequency of word i

negativeSample :: (MonadAsync m) => IntervalMap (StrictTuple Double Double) a -> AheadT m a
negativeSample im = S.concatMapM (const return ()) $ do
  (p :: Double) <- liftIO $ randomRIO (0, 1)
  pure . S.yield . snd . IvMap.findLast $ IvMap.containing im p

negativeSampleStream :: (MonadAsync m) => IntervalMap (StrictTuple Double Double) a -> AheadT m a
negativeSampleStream im = negativeSample im <> negativeSampleStream im

-- | Infinite stream of contiguous chunks of size `s`
shufflingStreamOfSize :: (MonadAsync m, Storable a) => Int -> SA.Array a -> AheadT m [a]
shufflingStreamOfSize s = S.chunksOf s FL.toList . (shufflingStreamOfSize' s)

-- | Infinite stream of samples from an array of some size
shufflingStreamOfSize' :: (MonadAsync m, Storable a) => Int -> SA.Array a -> AheadT m a
shufflingStreamOfSize' s a = sample s a <> shufflingStreamOfSize' s a

-- | Randomly samples a slice of size `size` from the array, returning it as a stream.
sample :: (MonadAsync m, Storable a) => Int -> SA.Array a -> AheadT m a
sample size a = S.concatMapM (const return ()) $ do
  start <- liftIO $ randomRIO (0, (SA.length a - size))
  pure . S.fromList $ fmap (SA.unsafeIndex a) [start .. (start + size)]

trainStream :: (MonadAsync m, Storable a, Ord a) => Int -> TokenStreamDataset a -> (AheadT m [a], AheadT m a)
trainStream i (TokenStreamDataset a mm im c) = (trainStream', negativeSampleStream im)
  where
    trainStream' = S.mapMaybeM (subsampleFilter c mm) $ shufflingStreamOfSize i a
