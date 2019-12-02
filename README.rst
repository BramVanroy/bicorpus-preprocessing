Bicorpus preprocessing
======================

This repository aims to preprocess a given parallel corpus in parallel so that it meets a specfic set of requirements.
The tool is capable of **removing** sentences

- that contain less than `n` and/or more than `m` tokens;
- where the ratio of number of tokens between source and target sentence is larger than expected;
- that are duplicates of other sentences in a white-space agnostic manner. This means that even if one sentence contains
  the same tokens as another one but more white-space characters, they will still be seen as duplicates. The first
  encountered duplicate will be kept;
- that are not written in an expected language (language detection happens through fasttext_ and/or are not given high
  enough predicted probability. Fasttext predicts the language of given input text and assigns a probability value.
  If the probability is not larger than or equal to a given value, a sentence can be discarded.

In addition, the output can be tokenized when requested. Pre/postprocessing scripts are available to either construct
a bicorpus from two separate parallel files, or to deconstruct a given bicorpus into separate files.

.. _fasttext: https://github.com/facebookresearch/fastText/tree/master/python

Installation
------------

Simply clone this repo and do a `pipenv install`. The only dependencies are `spacy` for tokenisation and `fasttext` for
language detection. Python 3.6 or higher is expected.

.. code-block:: bash

    git clone https://github.com/BramVanroy/bicorpus-preprocessing.git
    cd bicorpus-preprocessing
    pipenv install

Usage
-----

Do not forget to activate your virtual environment before usage, or use `pipenv run` instead.

.. code-block:: bash

    pipenv shell
    python main.py my_bicorpus.txt --src_lang en --tgt_lang nl
    # OR
    pipenv run python main.py my_bicorpus.txt --src_lang en --tgt_lang nl

Available arguments (as per `main.py -h`):

.. code-block:: bash

    usage: main.py [-h] [-s SRC_LANG] [--src_model SRC_MODEL] [-t TGT_LANG]
                   [--tgt_model TGT_MODEL] [--sep SEP] [-b BATCH_SIZE]
                   [--tokenize] [--dedupe] [--keep_order]
                   [--max_length MAX_LENGTH] [--min_length MIN_LENGTH]
                   [--max_ratio MAX_RATIO] [--min_prob MIN_PROB] [-n N_WORKERS]
                   fin

    Preprocess a bilingual corpus by verifying the language, checking min and max
    sentence length and the ratio between number of source and target tokens.
    Optionally tokenize and/or dedupe the output.

    positional arguments:
      fin                   Input file containing tab-separated source and target
                            sentences

    optional arguments:
      -h, --help            show this help message and exit
      -s SRC_LANG, --src_lang SRC_LANG
                            Source language (abbreviation) (default: en)
      --src_model SRC_MODEL
                            Source spaCy model (default: en_core_web_sm)
      -t TGT_LANG, --tgt_lang TGT_LANG
                            Target language (abbreviation) (default: nl)
      --tgt_model TGT_MODEL
                            Target spaCy model (default: nl_core_news_sm)
      --sep SEP             Separator to split sentences on (default: )
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Batch size (in kB) for the chunker (default: 1000)
      --tokenize            Tokenize the output (default: False)
      --dedupe              Deduplicate based on tokenized sentences, so
                            inconsistent whitespace should be handled fairly well
                            (default: False)
      --keep_order          Keep the same order of sentences as the source file.
                            Enabling this might be slower and consume more memory
                            when you have many/large batches. (default: False)
      --max_length MAX_LENGTH
                            Maximal number of tokens in a sentence that only
                            consist of alphanumeric characters (and not only
                            digits) (default: None)
      --min_length MIN_LENGTH
                            Minimal number of tokens in a sentence that only
                            consist of alphanumeric characters (and not only
                            digits) (default: None)
      --max_ratio MAX_RATIO
                            Maximal ratio of numbers of tokens in source and
                            target sentence (default: None)
      --min_prob MIN_PROB   The minimal certainty (or probability) for language
                            detection. If fasttext is less than 'min_prob' certain
                            about the predicted language, the sentence will be
                            discarded. (default: None)
      -n N_WORKERS, --n_workers N_WORKERS
                            Total number of workers (reader and writer processes
                            added on top of this number). Default depends on your
                            hardware (default: total_n_cpus-1)

The preprocessing script is called `bicorpus.py`. Its main arguments are `construct` and `deconstruct`. If you need
help using the script, just call `bicorpus.py construct -h` or equivalent.