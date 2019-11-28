import datetime
import logging
from math import inf
from multiprocessing import Manager, Process, Pool, cpu_count

import fasttext
import spacy
from tqdm import tqdm

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)


class Cleaner:
    def __init__(self, chunker, *,
                 src_lang='en',
                 tgt_lang='nl',
                 sep='\t',
                 tokenize=False,
                 dedupe=False,
                 src_model='en_core_web_sm',
                 tgt_model='nl_core_news_sm',
                 n_workers=cpu_count()-1,
                 max_length=None,
                 min_length=None,
                 max_ratio=None,
                 keep_order=False):
        self.dedupe = dedupe
        self.sep = sep
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenize = tokenize
        self.src_model = src_model
        self.tgt_model = tgt_model

        self.work_queue = None
        self.result_queue = None

        self.n_workers = n_workers
        self.chunker = chunker

        self.max_length = max_length if max_length is not None else inf
        self.min_length = min_length if min_length is not None else 0
        self.max_ratio = max_ratio

        self.keep_order = keep_order

        self.n_batches = 0

    def parse(self):
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
                jobs = [pool.apply_async(self.worker) for _ in range(self.n_workers)]

                for job in jobs:
                    _ = job.get()

            # clean-up
            reader_proc.join()
            reader_proc.terminate()

            self.result_queue.put('done')
            writer_proc.join()
            writer_proc.terminate()

        logging.info(f"Finished process in {datetime.datetime.now() - start_time}.")

    def worker(self):
        # ft_model can't be pickled
        # so init inside worker
        ft_model = fasttext.load_model('models/lid.176.bin')

        src_nlp = spacy.load(self.src_model, disable=['ner', 'textcat'])
        src_nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')
        tgt_nlp = spacy.load(self.tgt_model, disable=['ner', 'textcat'])
        tgt_nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')

        while True:
            # Get work from the working queue
            work = self.work_queue.get()
            if work == 'done':
                break

            chunk_start, chunk_size = work
            src_sentences, tgt_sentences = self.process_batch(chunk_start, chunk_size, ft_model, src_nlp, tgt_nlp)
            self.result_queue.put((chunk_start, chunk_size, src_sentences, tgt_sentences))

    def process_batch(self, chunk_start, chunk_size, ft_model, src_nlp=None, tgt_nlp=None, cb=None):
        batch = self.chunker.get_batch(chunk_start, chunk_size)
        src, tgt = zip(*[map(str.strip, l.split(self.sep, maxsplit=2)) for l in batch])

        src = self.lang_processor(src, self.src_lang, ft_model, src_nlp)
        tgt = self.lang_processor(tgt, self.tgt_lang, ft_model, tgt_nlp)

        invalid_idxs = {idx for idx, tupl in enumerate(src) if not tupl[2]}
        invalid_idxs.update({idx for idx, tupl in enumerate(tgt) if not tupl[2]})

        if invalid_idxs:
            src, tgt = self._delete_idxs(src, tgt, idxs=invalid_idxs)

        if self.max_ratio:
            src, tgt = self.check_ratio(src, tgt)

        # remove the 'invalid' bool: now list of tuples: (sent, tok_sent)
        src = [(s[0], s[1]) for s in src]
        tgt = [(s[0], s[1]) for s in tgt]

        if cb:
            src, tgt = cb(src, tgt)

        return src, tgt

    def lang_processor(self, batch, lang, ft_model, nlp=None):
        sentences = []
        docs = nlp.pipe(batch)
        for doc in docs:
            for sent in doc.sents:
                tokens = list(sent)
                # only count tokens that consist of alphanum chars and not only digits
                n_valid_tokens = len([t for t in tokens if t.text.isalnum() and not t.is_digit])
                len_valid = self.min_length <= n_valid_tokens <= self.max_length

                lang_label, _ = ft_model.predict(sent.text)
                lang_valid = lang_label[0].replace('__label__', '') == lang

                tokenized_sent = ' '.join([t.text for t in tokens])

                if self.tokenize:
                    sent = tokenized_sent
                else:
                    sent = sent.text

                sentences.append((sent, tokenized_sent, len_valid and lang_valid))

        return sentences

    def writer(self):
        pfin = self.chunker.pfin
        pf_src = pfin.with_suffix(f".{self.src_lang}")
        pf_tgt = pfin.with_suffix(f".{self.tgt_lang}")

        with pf_src.open('w', encoding='utf-8') as fh_src, pf_tgt.open('w', encoding='utf-8') as fh_tgt:
            n_sentences = self._write(fh_src, fh_tgt)

        n_removed = self.chunker.n_lines - n_sentences
        logging.info(f"Wrote {n_sentences:,} sentences (removed {n_removed:,}) to '{pf_src}' and '{pf_tgt}.")

    def _write(self, fh_src, fh_tgt):
        """ Results are not necessarily returned in order, so use prev_chunk_end
            to ensure the correct output order. """
        src_tok_sents = set()
        tgt_tok_sents = set()

        n_sentences = 0
        prev_chunk_end = 0
        pbar = tqdm(total=self.n_batches, desc='Progress', unit='batch')

        def _process(src, tgt, chnk_size=None):
            nonlocal pbar, n_sentences, prev_chunk_end, src_tok_sents, tgt_tok_sents

            if chnk_size:
                prev_chunk_end += chnk_size

            # Optionally filter duplicated sentences (tokenized)
            for src_tup, tgt_tup in zip(src, tgt):
                if not self.dedupe or (src_tup[1] not in src_tok_sents and tgt_tup[1] not in tgt_tok_sents):
                    fh_src.write(src_tup[0] + '\n')
                    fh_tgt.write(tgt_tup[1] + '\n')
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

    def reader(self):
        chunks = list(self.chunker.chunkify())
        self.result_queue.put(len(chunks))

        for chunk_tuple in chunks:
            self.work_queue.put(chunk_tuple)

        for _ in range(self.n_workers):
            self.work_queue.put('done')

    @staticmethod
    def _delete_idxs(*lists, idxs):
        for i in sorted(idxs, reverse=True):
            for l in lists:
                try:
                    del l[i]
                except IndexError:
                    raise IndexError(f"index {i} in list {l}")
        return lists

    @staticmethod
    def _prevent_sbd(doc):
        # If you already have one sentence per line in your file
        # you may wish to disable sentence segmentation with this function,
        # which is added to the nlp pipe in the constructor
        for token in doc:
            token.is_sent_start = False
        return doc

    def check_ratio(self, src, tgt):
        valid_src = []
        valid_tgt = []
        for src_tuple, tgt_tuple in zip(src, tgt):
            if len(src_tuple[1]) / len(tgt_tuple[1]) > self.max_ratio:
                continue
            valid_src.append(src_tuple)
            valid_tgt.append(tgt_tuple)

        return valid_src, valid_tgt
