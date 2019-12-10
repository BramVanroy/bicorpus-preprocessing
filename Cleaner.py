import datetime
import logging
from math import inf
from multiprocessing import cpu_count, Manager, Pool, Process
from typing import Iterable, List, Optional, TextIO, Tuple

import fasttext
import spacy
from tqdm import tqdm

from Chunker import Chunker

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)


class Cleaner:
    """ Creates a cleaner object which has parse() as its main entrypoint. It can clean a bicorpus by
        performing operations such as deduplicating (by tokenized sentences), restrict the length of
        sentences, only keeping text where a language predictor (fasttext) is certain of the language.
        The system can optionally also tokenize the output files.

        If required, output files can be merged together with the utility script in bicorpus.py.

    :param chunker: a chunker instance that chunks the input file for high-performance parallel processing
    :param dedupe: whether to deduplicate the text. Deduplicating is done based on tokenized sentences rather
                   than actual sentences to avoid seemingly non-duplicates to exist.
    :param keep_order: whether to maintain the order of the input file. Might be slower and consume more memory.
    :param max_length: an optional max length. Sentences that are longer will not be included.
    :param max_ratio: an optional max ratio between the source and target sentence. Lines for which the ratio
                      is larger will not be included.
    :param min_length: an optional min length. Sentences that are shorter will not be included.
    :param min_prob: an optional minimal probability. If the detector (fasttext) shows a lower probability,
                     the sentence will not be included. If left empty, the only condition is that the highest
                     predicted language is the same as the requested language.
    :param n_workers: how many workers to use for parallel processing.
    :param sep: separator to separate the input lines on, resulting in the source and target text.
    :param src_lang: abbreviation of the source language (e.g. 'en')
    :param src_model: spaCy model to use (e.g. 'en_core_web_sm')
    :param tgt_model: abbreviation of the target language (e.g. 'nl')
    :param tgt_lang: spaCy model to use (e.g. 'nl_core_news_sm')
    :param tokenize: whether to save the output as tokenized sentences.
    """
    def __init__(self,
                 chunker: Chunker,
                 *,
                 dedupe: bool = False,
                 do_lower_case: bool = False,
                 keep_order: bool = False,
                 max_length: Optional[int] = None,
                 max_ratio: Optional[int] = None,
                 min_length: Optional[int] = None,
                 min_prob: Optional[int] = None,
                 n_workers: int = cpu_count() - 1,
                 sep: str = '\t',
                 src_lang: str = 'en',
                 src_model: str = 'en_core_web_sm',
                 tgt_model: str = 'nl_core_news_sm',
                 tgt_lang: str = 'nl',
                 tokenize: bool = False):
        self.chunker = chunker

        self.dedupe = dedupe
        self.do_lower_case = do_lower_case
        self.keep_order = keep_order
        self.max_length = max_length if max_length is not None else inf
        self.max_ratio = max_ratio
        self.min_length = min_length if min_length is not None else 0
        self.min_prob = min_prob
        self.n_workers = n_workers
        self.sep = sep
        self.src_lang = src_lang
        self.src_model = src_model
        self.tgt_lang = tgt_lang
        self.tgt_model = tgt_model
        self.tokenize = tokenize

        self.result_queue = None
        self.work_queue = None

        self.n_batches: int = 0

    def parse(self):
        """ Parses the input file that is chunked by the chunker in parallel.
            Writes only the sentences for which all conditions hold true, optionally tokenized.
        """
        start_time = datetime.datetime.now()

        with Manager() as manager:
            self.work_queue = manager.Queue()
            self.result_queue = manager.Queue()

            # The reader starts filling up the work_queue
            reader_proc = Process(target=self.reader)
            reader_proc.start()

            # block until writer has returned n_batches
            self.n_batches = self.result_queue.get()

            writer_proc = Process(target=self.writer)
            writer_proc.start()

            with Pool(processes=self.n_workers) as pool:
                jobs = [pool.apply_async(self._init_worker) for _ in range(self.n_workers)]

                for job in jobs:
                    _ = job.get()

            # clean-up
            reader_proc.join()
            reader_proc.terminate()

            self.result_queue.put('done')
            writer_proc.join()
            writer_proc.terminate()

        logging.info(f"Finished process in {datetime.datetime.now() - start_time}.")

    def _check_ratio(self,
                     src: List[Tuple],
                     tgt: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
        """ Check whether the ratio of number of tokens between source and target sentence does not
            exceed a given max value.

        :param src: list of tuples containing a sentence, tokenized sentence, boolean indicating
                    whether or not this is a valid sentence.
        :param tgt: list of tuples containing a sentence, tokenized sentence, boolean indicating
                    whether or not this is a valid sentence.
        :return: tuple of valid source and target sentences (each a list of tuples)
        """
        valid_src = []
        valid_tgt = []
        for src_tuple, tgt_tuple in zip(src, tgt):
            if len(src_tuple[1]) / len(tgt_tuple[1]) > self.max_ratio:
                continue
            valid_src.append(src_tuple)
            valid_tgt.append(tgt_tuple)

        return valid_src, valid_tgt

    def _init_worker(self):
        """ Initializes a worker. Each worker reads from the queue, processes
            a batch, and puts the output in another queue. """
        # ft_model can't be pickled
        # so init inside worker
        ft_model = fasttext.load_model('models/lid.176.bin')
        src_nlp = spacy.load(self.src_model, disable=['parser', 'tagger', 'ner', 'textcat'])
        tgt_nlp = spacy.load(self.tgt_model, disable=['parser', 'tagger', 'ner', 'textcat'])
        while True:
            # Get work from the working queue
            work = self.work_queue.get()
            if work == 'done':
                break

            chunk_start, chunk_size = work
            src_sentences, tgt_sentences = self._process_batch(chunk_start, chunk_size, ft_model, src_nlp, tgt_nlp)
            self.result_queue.put((chunk_start, chunk_size, src_sentences, tgt_sentences))

    def _lang_processor(self,
                        batch: Iterable[str],
                        lang: str,
                        ft_model: fasttext.FastText._FastText,
                        nlp: spacy.lang) -> List[Tuple[str, str, bool]]:
        """ Does the main language checking. Firstly, tokenizes the input. Then it checks whether sentences
            are in the expected language, optionally with a minimal probability. Finally, also optionally,
            the minimal and maximal number of tokens are verified.

        :param batch: batch of sentences to process (a list of strings)
        :param lang: the abbreviation of the language to use (e.g. 'en')
        :param ft_model: the initialized fasttext model
        :param nlp: the inititialized spaCy model
        :return: a list of tuples. Each tuple consists of the source sentence, tokenized sentence,
                 and a boolean indicating whether or not the sentence passed all conditions.
        """
        sentences = []
        docs = nlp.pipe(batch)
        for doc in docs:
            tokens = list(doc)
            # only count tokens that consist of alphanum chars and not only digits
            n_valid_tokens = len([t for t in tokens if t.text.isalnum() and not t.text.isdigit()])
            len_valid = self.min_length <= n_valid_tokens <= self.max_length
            tokenized_sent = ' '.join([t.text for t in tokens])

            lang_valid = True
            if len_valid:
                # fasttext expects preprocessed text (it splits on spaces)
                # see https://github.com/facebookresearch/fastText/tree/master/python
                lang_label, prob = ft_model.predict(tokenized_sent)

                # check if; if not: invalid
                # 1. the predicted language is the one we expected
                # 2. the probability is as high as we expected
                if (lang_label[0].replace('__label__', '') != lang) \
                        or (self.min_prob is not None and prob.item(0) < self.min_prob):
                    lang_valid = False

            valid = len_valid and lang_valid

            sentences.append((doc.text, tokenized_sent, valid))

        return sentences

    def _process_batch(self,
                       chunk_start: int,
                       chunk_size: int,
                       ft_model: fasttext.FastText._FastText,
                       src_nlp: spacy.lang,
                       tgt_nlp: spacy.lang) -> Tuple[List[Tuple[str, str]], ...]:
        """ The overarching process to control the processing of a single batch. The batch still needs to
            be retrieved, given a chunk_start and chunk_size argument.

        :param chunk_start: start byte for this batch
        :param chunk_size: size (in bytes) of this batch
        :param ft_model: the initialized fasttext model
        :param src_nlp: the inititialized spaCy model for the source language
        :param tgt_nlp: the inititialized spaCy model for the source language
        :return: a tuple consisting of the source and target sentences. Each item is a list of
                 tuples, consisting of the source sentence and the tokenized sentence
        """
        batch = self.chunker.get_batch(chunk_start, chunk_size)
        src, tgt = zip(*[map(str.strip, l.split(self.sep, maxsplit=2)) for l in batch])

        src = self._lang_processor(src, self.src_lang, ft_model, src_nlp)
        tgt = self._lang_processor(tgt, self.tgt_lang, ft_model, tgt_nlp)

        invalid_idxs = {idx for idx, tupl in enumerate(src) if not tupl[2]}
        invalid_idxs.update({idx for idx, tupl in enumerate(tgt) if not tupl[2]})

        if invalid_idxs:
            src, tgt = self._delete_idxs(src, tgt, idxs=invalid_idxs)

        if self.max_ratio:
            src, tgt = self._check_ratio(src, tgt)

        # remove the 'invalid' bool: now list of tuples: (sent, tok_sent)
        if self.do_lower_case:
            src = [(s[0].lower(), s[1].lower()) for s in src]
            tgt = [(s[0].lower(), s[1].lower()) for s in tgt]
        else:
            src = [(s[0], s[1]) for s in src]
            tgt = [(s[0], s[1]) for s in tgt]

        return src, tgt

    @staticmethod
    def _delete_idxs(src: List[Tuple],
                     tgt: List[Tuple],
                     idxs: Iterable[int]) -> Tuple[List[Tuple], List[Tuple]]:
        """ Deletes indices from two given lists (src and tgt)

        :param src: the list containing source elements
        :param tgt: the list containing target elements
        :param idxs: the indices to remove
        :return: a tuple consisting of the src and tgt lists with the given idxs removed
        """
        for i in sorted(idxs, reverse=True):
            for l in (src, tgt):
                try:
                    del l[i]
                except IndexError:
                    raise IndexError(f"Could not remove index {i} in list {l}")
        return src, tgt

    # READER/WRITER METHODS
    def reader(self):
        """ A reader function, ideally connected to a separate process. Reads all chunks
            from the chunker. Puts data in the work_queue. First the number of batches
            (processed in main process), then all batches, one-by-one.
        """
        # first get the number of chunks (batches)
        # this is used in tqdm to keep track of processed batches
        chunks = list(self.chunker.chunkify())
        self.result_queue.put(len(chunks))

        for chunk_tuple in chunks:
            self.work_queue.put(chunk_tuple)

        for _ in range(self.n_workers):
            self.work_queue.put('done')

    def writer(self):
        """ A writer function, ideally connected to a separate process. Opens file
            streams and delegates all the actual writing to _write. """
        pfin = self.chunker.pfin
        tok_suff = '.tok' if self.tokenize else ''
        tok_suff += '.low' if self.do_lower_case else ''
        pf_src = pfin.with_suffix(f"{tok_suff}.{self.src_lang}")
        pf_tgt = pfin.with_suffix(f"{tok_suff}.{self.tgt_lang}")

        with pf_src.open('w', encoding='utf-8') as fh_src, pf_tgt.open('w', encoding='utf-8') as fh_tgt:
            n_sentences = self._write(fh_src, fh_tgt)

        n_removed = self.chunker.n_lines - n_sentences
        logging.info(f"Wrote {n_sentences:,} sentences (removed {n_removed:,}) to '{pf_src}' and '{pf_tgt}.")

    def _write(self, fh_src: TextIO, fh_tgt: TextIO):
        """ Reads chunks from the result_queue, and write them to the given file objects.

        :param fh_src: Open file handle for the source file.
        :param fh_tgt: Open file handle for the traget file.
        """
        src_tok_sents = set()
        tgt_tok_sents = set()

        n_sentences = 0
        prev_chunk_end = 0
        pbar = tqdm(total=self.n_batches, desc='Progress', unit='batch')

        def _process(src: Tuple[str, str], tgt: Tuple[str, str], chnk_size: Optional[int] = None):
            nonlocal pbar, n_sentences, prev_chunk_end, src_tok_sents, tgt_tok_sents

            if chnk_size:
                prev_chunk_end += chnk_size

            # Optionally filter duplicated sentences (tokenized)
            for src_tup, tgt_tup in zip(src, tgt):
                if not self.dedupe or (src_tup[1] not in src_tok_sents and tgt_tup[1] not in tgt_tok_sents):
                    src_sent = src_tup[1] if self.tokenize else src_tup[0]
                    tgt_sent = tgt_tup[1] if self.tokenize else tgt_tup[0]

                    fh_src.write(f"{src_sent}\n")
                    fh_tgt.write(f"{tgt_sent}\n")
                    n_sentences += 1

                    if self.dedupe:
                        src_tok_sents.add(src_tup[1])
                        tgt_tok_sents.add(tgt_tup[1])
            pbar.update(1)

        results = {}
        while True:
            work = self.result_queue.get()
            if work == 'done':
                break
            chunk_start, chunk_size, src_sentences, tgt_sentences = work

            if self.keep_order:
                # process all examples in order
                if prev_chunk_end == chunk_start:
                    _process(src_sentences, tgt_sentences, chunk_size)

                    # check if existing data in the results follows
                    # the newly added data
                    if results:
                        while True:
                            nxt = results.pop(prev_chunk_end, None)
                            if nxt:
                                _process(nxt['src'], nxt['tgt'], nxt['chunk_size'])
                            else:
                                break
                else:
                    results[chunk_start] = {
                        'src': src_sentences,
                        'tgt': tgt_sentences,
                        'chunk_size': chunk_size
                    }
            else:
                _process(src_sentences, tgt_sentences)

        pbar.close()
        return n_sentences
