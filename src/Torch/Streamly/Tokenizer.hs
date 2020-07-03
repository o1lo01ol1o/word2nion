{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE OverloadedStrings #-}
module Torch.Streamly.Tokenizer
  ( Token (..),
    getTokens,
    tokenize
  )
where

import Data.Char
import qualified Data.List as List
import Data.Text
import Prelude hiding (dropWhile, head, null, tail, takeWhile)

-- Slightly adapted from: https://hackage.haskell.org/package/glider-nlp-0.4/docs/src/Glider-NLP-Tokenizer.html#tokenize

-- | Token type
data Token
  = Token Text
  | Number Text
  | Punctuation Char
  | Symbol Char
  | Whitespace
  | Unknown Char
  deriving stock (Eq, Show)

-- | Split text into tokens
--
-- > tokenize "one two." == [Word "one", Whitespace, Word "two", "Separator "."]
tokenize :: Text -> [Token]
tokenize xs = case allParser xs of
  [(v, out)] -> v : tokenize out
  _ -> []

-- | Extract all words from tokens
--
-- > getTokens "one two." == ["one", "two"]
getTokens :: [Token] -> [Text]
getTokens [] = []
getTokens (x : xs) = case x of
  Token a -> a : getTokens xs
  _ -> getTokens xs

-- | Convert all words to the same case
foldCase :: [Text] -> [Text]
foldCase = List.map toCaseFold

-- | Parser type
type Parser = Text -> [(Token, Text)]

-- | Parse word
wordParser :: Parser
wordParser xs
  | null xs = []
  | isLetter (head xs) = [(Token (takeWhile isAlphaNum xs), dropWhile isAlphaNum xs)]
  | otherwise = []

-- | Parse number
numberParser :: Parser
numberParser xs
  | null xs = []
  | isDigit (head xs) = [(Number (takeWhile isDigit xs), dropWhile isDigit xs)]
  | otherwise = []

-- | Parse punctuation
punctuationParser :: Parser
punctuationParser xs
  | null xs = []
  | isPunctuation (head xs) = [(Punctuation (head xs), tail xs)]
  | otherwise = []

-- | Parse symbol
symbolParser :: Parser
symbolParser xs
  | null xs = []
  | isSymbol (head xs) = [(Symbol (head xs), tail xs)]
  | otherwise = []

-- | Parse whitespaces
spaceParser :: Parser
spaceParser xs
  | null xs = []
  | isSpace (head xs) = [(Whitespace, xs)]
  | otherwise = []

-- | Parse single char
charParser :: Parser
charParser xs
  | null xs = []
  | otherwise = [(Unknown (head xs), tail xs)]

-- | Apply all parsers to the input.
-- Return result from the first which will parse correctly given text.
allParser :: Parser
allParser xs = case wordParser xs of
  [(v, out)] -> [(v, out)]
  _ -> case numberParser xs of
    [(v, out)] -> [(v, out)]
    _ -> case punctuationParser xs of
      [(v, out)] -> [(v, out)]
      _ -> case symbolParser xs of
        [(v, out)] -> [(v, out)]
        _ -> case spaceParser xs of
          [(v, out)] -> [(v, out)]
          _ -> charParser xs
