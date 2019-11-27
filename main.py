import argparse
from multiprocessing import cpu_count

from Chunker import Chunker
from Cleaner import Cleaner

DESC = '''Preprocess a bilingual corpus by verifying the language, checking min and max sentence length
          and the ratio between number of source and target tokens. Optionally tokenize the output.'''

if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description=DESC,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument('fin', help='Input file containing tab-separated source and target sentences')
    cparser.add_argument('-s', '--src_lang', help='Source language (abbreviation)', default='en')
    cparser.add_argument('--src_model', help='Source spaCy model', default='en_core_web_sm')
    cparser.add_argument('-t', '--tgt_lang', help='Target language (abbreviation)', default='nl')
    cparser.add_argument('--tgt_model', help='Target spaCy model', default='nl_core_news_sm')
    cparser.add_argument('-b', '--batch_size', help='Batch size (in kB) for the chunker', default=1000)
    cparser.add_argument('--tokenize', action='store_true', help='Tokenize the output')
    cparser.add_argument('--max_length', help='Maximal number of tokens in a sentence that only consist of'
                                              ' alphanumeric characters (and not only digits)')
    cparser.add_argument('--min_length', help='Minimal number of tokens in a sentence that only consist of'
                                              ' alphanumeric characters (and not only digits)')
    cparser.add_argument('--max_ratio', help='Maximal ratio of numbers of tokens in source and target sentence')
    cparser.add_argument('-n', '--n_workers', help='Total number of workers'
                                                   ' (reader and writer processes added on top of this number).'
                                                   ' Default depends on your hardware', default=cpu_count()-1)

    cargs = cparser.parse_args()
    cargs = vars(cargs)

    chunkr = Chunker(cargs.pop('fin'), batch_size=cargs.pop('batch_size'))
    cleanr = Cleaner(chunkr, **cargs)
    cleanr.parse()
