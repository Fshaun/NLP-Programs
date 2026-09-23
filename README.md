# NLP Programs

This repository contains short NLP scripts demonstrating core text-processing and automata concepts.

## nd_recognize.py

`nd_recognize.py` implements a non-deterministic finite-state automaton (NFSA) recognizer for a small subset of regular expressions.

It does the following:
- Builds an automaton from a regex-like pattern such as `baaa!` or `baa*!`
- Supports literal characters and a trailing `*` operator on the preceding character
- Uses epsilon transitions to represent optional repeated symbols
- Runs the automaton against an input string using an agenda-based search
- Can search with either depth-first or breadth-first strategy
- Prints the search steps and whether the string is accepted or rejected

In other words, this script is a small textbook implementation of the ND-RECOGNIZE algorithm from speech and language processing, showing how a regex-like pattern can be compiled into an automaton and then matched against text.

## preprocessing_pipeline.py

`preprocessing_pipeline.py` demonstrates a simple text preprocessing pipeline for clinical text.

It does the following:
- Takes raw text and lowercases it
- Tokenizes using a custom regular-expression based tokenizer
- Preserves hyphenated clinical terms such as `intensive-care` as single tokens
- Removes stopwords using a custom stopword list
- Lemmatizes tokens using NLTK's `WordNetLemmatizer` with POS-aware tagging
- Prints the result after each stage so the transformation can be inspected step by step

This script is useful for showing how raw text can be cleaned and normalized before further NLP tasks such as classification or feature extraction.

## Summary

- `nd_recognize.py` = automata/regex matching logic
- `preprocessing_pipeline.py` = text cleaning and normalization for downstream NLP work
