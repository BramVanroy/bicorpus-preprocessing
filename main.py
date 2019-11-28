import argparse
from multiprocessing import cpu_count

from Chunker import Chunker
from Cleaner import Cleaner

DESC = '''Preprocess a bilingual corpus by verifying the language, checking min and max sentence length
          and the ratio between number of source and target tokens. Optionally tokenize and/or dedupe the output.'''

if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description=DESC,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument('fin', help='Input file containing tab-separated source and target sentences')
    cparser.add_argument('-s', '--src_lang', help='Source language (abbreviation)', default='en')
    cparser.add_argument('--src_model', help='Source spaCy model', default='en_core_web_sm')
    cparser.add_argument('-t', '--tgt_lang', help='Target language (abbreviation)', default='nl')
    cparser.add_argument('--tgt_model', help='Target spaCy model', default='nl_core_news_sm')
    cparser.add_argument('--sep', help='Separator to split sentences on', default='\t')
    cparser.add_argument('-b', '--batch_size', help='Batch size (in kB) for the chunker', default=1000, type=int)
    cparser.add_argument('--tokenize', action='store_true', help='Tokenize the output')
    cparser.add_argument('--dedupe',
                         action='store_true',
                         help='Deduplicate based on tokenized sentences, so inconsistent'
                              ' whitespace should be handled fairly well')
    cparser.add_argument('--keep_order',
                         action='store_true',
                         help='Keep the same order of sentences as the source file. Enabling this might'
                              ' be slower and consume more memory when you have many/large batches.')
    cparser.add_argument('--max_length',
                         help='Maximal number of tokens in a sentence that only consist of alphanumeric'
                              ' characters (and not only digits)',
                         type=int)
    cparser.add_argument('--min_length',
                         help='Minimal number of tokens in a sentence that only consist of alphanumeric'
                              ' characters (and not only digits)',
                         type=int)
    cparser.add_argument('--max_ratio',
                         help='Maximal ratio of numbers of tokens in source and target sentence',
                         type=int)
    cparser.add_argument('-n', '--n_workers',
                         help='Total number of workers (reader and writer processes added on top of this number).'
                              ' Default depends on your hardware',
                         default=cpu_count()-1,
                         type=int)

    cargs = cparser.parse_args()
    cargs = vars(cargs)

    chunkr = Chunker(cargs.pop('fin'), batch_size=cargs.pop('batch_size'))
    cleanr = Cleaner(chunkr, **cargs)
    cleanr.parse()
