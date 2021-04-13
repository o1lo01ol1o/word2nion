# word2nion

[![GitHub CI](https://github.com/o1lo01ol1o/word2nion/workflows/CI/badge.svg)](https://github.com/o1lo01ol1o/word2nion/actions)
[![Hackage](https://img.shields.io/hackage/v/word2nion.svg?logo=haskell)](https://hackage.haskell.org/package/word2nion)

A handful of small experiments on NLP and (non-)commutive algebras.


# Dev

## Hoogle

```console
./scripts/hoogle
```

and go to <https://localhost:8080>

## GHCI

```console
nix-shell --run "cabal new-repl"
```

This loads the pragmas and impots set in `.ghci` in addition to the cabal-specified local library.

## Building and running

eg:

```bash
nix-shell
cabal new-build lib:word2nion 
cabal new-build exe:word2nion 
```

Then execute the binary at the listed path.

# Current roadmap

## Sanity tests and initial exploration

- [x] Quaternion-based version of word2vec
- [x] word2vec (skip-gram & negative sampling) 
- [ ] make sense of any differences in behavior
- [ ] UMAP plot the two embedding spaces
- [ ] Word2FinVec implementation

## higher Caley-dickson iterations

- [ ] generalize the quaternion implmentation to generate an algebra for any `N`ion

## Transformer

- [ ] replace products with `N`ion algebra
- [ ] profit?
