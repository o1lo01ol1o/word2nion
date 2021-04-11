{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -O2 #-}
-- | Helpers for tokenizing text.
-- Slightly adapted from: https://hackage.haskell.org/package/glider-nlp-0.4/docs/src/Glider-NLP-Tokenizer.html#tokenize
-- FIXME: Attoparser doesn't liek recursion.  rewrite in attoparsec
module Torch.Streamly.Tokenizer
  ( Token (..),
    getTokens,
    tokenize,
  )
where

import Control.Applicative (Alternative ((<|>)))
import Data.Attoparsec.ByteString.Char8 as AttoChar
  ( Parser,
    anyChar,
    double,
    isDigit,
    many1',
    parseOnly,
    satisfy,
    space,
    takeWhile1, endOfInput, many'
  )
import Data.ByteString (ByteString)
import Data.Char (isAlpha, isPunctuation, isSymbol)
import Prelude
import qualified Data.Attoparsec as Atto
import Data.Word (Word8)

-- | Token type
data Token
  = Token ByteString
  | Number (Either ByteString Double)
  | Punctuation Char
  | Symbol Char
  | Whitespace
  | Unknown Word8
  deriving stock (Eq, Ord, Show)

-- | Split text into tokens or return an error string from attoparsec.
--
-- >>> tokenize "The year was 1984, and the world was a better place."
-- >>> Right [Token "The",Whitespace,Token "year",Whitespace,Token "was",Whitespace,Number (Right 1984.0),Punctuation ',',Whitespace,Token "and",Whitespace,Token "the",Whitespace,Token "world",Whitespace,Token "was",Whitespace,Token "a",Whitespace,Token "better",Whitespace,Token "place",Punctuation '.']
tokenize :: ByteString -> Either String [Token]
tokenize = AttoChar.parseOnly (allParser <* endOfInput)

-- | Extract all words from tokens
--
-- >>> getTokens [Token "one", Whitespace, Token "two", "Separator "."]
-- >>> ["one", "two"]
getTokens :: [Token] -> [ByteString]
getTokens [] = []
getTokens (x : xs) = case x of
  Token a -> a : getTokens xs
  _ -> getTokens xs

-- | Parse word
wordParser :: AttoChar.Parser Token
wordParser = Token <$> AttoChar.takeWhile1 isAlpha

-- | Parse number
numberParser :: AttoChar.Parser Token
numberParser = doubleParser

doubleParser :: AttoChar.Parser Token
doubleParser = Number . Right <$> AttoChar.double

-- | Parse punctuation
punctuationParser :: AttoChar.Parser Token
punctuationParser = Punctuation <$> AttoChar.satisfy isPunctuation

-- | Parse symbol
symbolParser :: AttoChar.Parser Token
symbolParser = Symbol <$> AttoChar.satisfy isSymbol

-- | Parse whitespaces
spaceParser :: AttoChar.Parser Token
spaceParser = Whitespace <$ AttoChar.space

-- | Parse single char
charParser :: AttoChar.Parser Token
charParser = Unknown <$> Atto.anyWord8

-- | Parse the '<unk>' tag
unkParser :: AttoChar.Parser Token
unkParser = Token <$> do 
  _l <- AttoChar.satisfy $ (==) '<'
  _u <- AttoChar.satisfy $ (==) 'u'
  _n <- AttoChar.satisfy $ (==) 'n'
  _k <- AttoChar.satisfy $ (==) 'k'
  _r <- AttoChar.satisfy $ (==) '>'
  pure "<unk>"

-- | Apply all parsers to the input.
-- Return result from the first which will parse correctly given text.
allParser :: AttoChar.Parser [Token]
allParser = AttoChar.many' eachParser

eachParser :: AttoChar.Parser Token
eachParser =
  wordParser 
    <|> unkParser
    <|> numberParser
    <|> punctuationParser
    <|> symbolParser
    <|> spaceParser
    <|> charParser
