{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE ConstraintKinds            #-}
{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveAnyClass             #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE DerivingStrategies         #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedLabels           #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE StandaloneDeriving         #-}
{-# LANGUAGE Strict                     #-}
{-# LANGUAGE StrictData                 #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TypeApplications           #-}
{-# LANGUAGE TypeFamilies               #-}
{-# LANGUAGE UndecidableInstances       #-}

module Torch.Streamly.Dataloader
  ( TokenStreamDataset (..),
    dataset,
    trainStream,
    byTwos
  )
where

import           Control.Monad.Catch                   (MonadCatch)
import           Control.Monad.IO.Class                (MonadIO, liftIO)
import           Data.Bifunctor                        (bimap)
import           Data.IntervalMap.Generic.Strict       (IntervalMap)
import qualified Data.IntervalMap.Generic.Strict       as IM
import qualified Data.IntervalMap.Generic.Strict       as IvMap
import           Data.List                             (sortOn)
import           Data.Map.Monoidal.Strict              (MonoidalMap)
import qualified Data.Map.Monoidal.Strict              as MMap
import qualified Data.Maybe                            as Maybe (mapMaybe)
import           Data.Monoid                           (First (..), Sum (..))
import qualified Data.Set                              as Set
import qualified Data.Text                             as T
import           Foreign.Storable
import           GHC.Generics                          (Generic)
import           Path
import           Streamly
import qualified Streamly.Data.Fold                    as FL
import qualified Streamly.Internal.Data.Fold           as FL
import           Streamly.Internal.Data.Unicode.Stream (decodeUtf8)
import qualified Streamly.Internal.FileSystem.File     as File
import qualified Streamly.Internal.Memory.Array        as SA
import           Streamly.Prelude                      as S hiding (minimum,
                                                             tail)
import           System.Random                         (randomRIO)
import           Torch.Streamly.Tokenizer

-- | Represents a dataset of some token type `a`
data TokenStreamDataset a = TokenStreamDataset
  { -- | The lookup map of tokens to integers
    tokenStreamDatasetToken :: MonoidalMap a (First Int),
    -- | The buffered, tokenized corpora in a streamly Array for O(1) random access
    tokenStreamDatasetData :: SA.Array Int,
    -- | The counts of the tokens in the corpora
    tokenStreamDatasetFrequencies :: MonoidalMap a (Sum Int),
    -- | The probability intervals per token (for sampling from the noise distribution)
    tokenStreamDatasetProbabilities :: IntervalMap (StrictTuple Double Double) Int,
    -- | The total number of examples
    tokenStreamDatasetOccurances :: Int
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

makeLookup' :: (Ord k) => MonoidalMap k a -> MonoidalMap k (First Int)
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

tokenStream :: (MonadAsync m, MonadCatch m) => Path a File -> AheadT m Token
tokenStream fp =
  S.concatMap (S.fromList . tokenize . T.pack)
    . S.splitOnSuffix (`Set.member` splitChars) FL.toList
    . decodeUtf8
    $ File.toBytes (toFilePath fp)

{-# INLINE splitChars #-}
splitChars :: Set.Set Char
splitChars = s
  where
    !s = Set.fromList ['.', '!', '?']

newtype SymbolOrToken = SymbolOrToken {unSymbolOrToken :: Either Char T.Text}
  deriving stock (Eq, Ord, Generic)

filterAndLowercase :: (MonadAsync m) => AheadT m Token -> AheadT m (Either Char T.Text)
filterAndLowercase = S.mapMaybe go
  where
    go (Token t)  = Just . Right $ T.toLower t
    go (Symbol t) = Just . Left $ t
    go _          = Nothing

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

tokenFrequencies :: (MonadAsync m, Ord a) => AheadT m a -> m (MonoidalMap a (Sum Int))
tokenFrequencies = S.fold (groupBy Just (const $ Sum 1)) . serially . aheadly

{-# INLINE byTwos #-}
byTwos :: [b] -> [(b, b)]
byTwos xs = zip xs $ tail xs

toIvMap :: Int -> MonoidalMap a (Sum Int) -> IntervalMap (StrictTuple Double Double) a
toIvMap total mm =
  IvMap.fromList . Maybe.mapMaybe go . byTwos $ xs
  where
    xs = bimap Just ((/ fromIntegral total) . fromIntegral . getSum) <$> sortOn snd (MMap.toAscList mm)
    go ((Just a, lb), (_, ub)) = Just $ ((StrictTuple lb ub), a)
    go _                       = Nothing

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
-- in the corpora. The original paper used the same caclulation as below on
-- the input word itself.  Since we're considering lists of words, we use the minimum value
-- of all the words in the list since we want to increase the chance that rare words are retained.
subsampleFilter :: (MonadIO m, Ord a) => Int -> MonoidalMap a (Sum Int) -> [a] -> m (Maybe [a])
subsampleFilter total mm d = do
  (p :: Double) <- liftIO $ randomRIO (0, 1)
  pure $ case (minimum $ calc mm <$> d) > p of
    True  -> Nothing -- Higer values of calc should be higher values of discarding, so if p is less than that value, discard.
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

-- | Make a training stream of token indicies of size windowSize and a negative
-- sampling stream from a `TokenStreamDataset a`
trainStream :: (MonadAsync m, Ord a) => Int -> TokenStreamDataset a -> (AheadT m [Int], AheadT m Int)
trainStream windowSize (TokenStreamDataset indexLookup a mm im c) = (trainStream', negativeSampleStream im)
  where
    trainStream' = S.mapMaybeM (subsampleFilter c (MMap.mapKeys (fromJust . getFirst . (indexLookup MMap.!)) mm)) $ shufflingStreamOfSize windowSize a
    fromJust (Just v) = v
    fromJust _ = error "trainStream: it is impossible that fromJust is partial here!  Something is wrong!"
