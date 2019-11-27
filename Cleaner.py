import datetime
import logging
from math import inf
from multiprocessing import Manager, Process, Pool, cpu_count

import fasttext
import spacy

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)


class Cleaner:
    def __init__(self, chunker, *,
                 src_lang='en',
                 tgt_lang='nl',
                 tokenize=False,
                 src_model='en_core_web_sm',
                 tgt_model='nl_core_news_sm',
                 n_workers=cpu_count()-1,
                 max_length=None,
                 min_length=None,
                 max_ratio=None):
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

    def parse(self):
        start_time = datetime.datetime.now()

        with Manager() as manager:
            self.work_queue = manager.Queue()
            self.result_queue = manager.Queue()

            # The reader starts filling up the work_queue
            reader_proc = Process(target=self.reader)
            reader_proc.start()

            writer_proc = Process(target=self.writer)
            writer_proc.start()

            total_init_sents = 0
            total_final_sents = 0
            with Pool(processes=self.n_workers) as pool:
                jobs = [pool.apply_async(self.worker) for _ in range(self.n_workers)]

                for job_idx, job in enumerate(jobs, 1):
                    # get initial batch_size and final number of sents
                    n_init_sents, n_final_sents = job.get()
                    total_init_sents += n_init_sents
                    total_final_sents += n_final_sents

            # clean-up
            reader_proc.join()
            reader_proc.terminate()

            self.result_queue.put('done')
            writer_proc.join()
            writer_proc.terminate()

        logging.info(f"Finished process in {datetime.datetime.now() - start_time}."
                     f" Removed {(total_init_sents-total_final_sents):,} out of {total_init_sents:,} sentences.")

    def worker(self):
        ft_model = fasttext.load_model('models/lid.176.bin')

        src_nlp = spacy.load(self.src_model, disable=['ner', 'textcat'])
        src_nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')
        tgt_nlp = spacy.load(self.tgt_model, disable=['ner', 'textcat'])
        tgt_nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')

        total_init_sents = 0
        total_final_sents = 0
        while True:
            # Get work from the working queue
            work = self.work_queue.get()
            if work == 'done':
                break

            chunk_start, chunk_size = work
            src_sentences, tgt_sentences, batch_size, final_size = self.process_batch(chunk_start,
                                                                                      chunk_size,
                                                                                      ft_model,
                                                                                      src_nlp,
                                                                                      tgt_nlp)
            self.result_queue.put((chunk_start, chunk_size, src_sentences, tgt_sentences))

            total_init_sents += batch_size
            total_final_sents += final_size

        return total_init_sents, total_final_sents

    def process_batch(self, chunk_start, chunk_size, ft_model, src_nlp=None, tgt_nlp=None):
        batch = self.chunker.get_batch(chunk_start, chunk_size)
        src, tgt = zip(*[map(str.strip, l.split('\t')) for l in batch])
        n_init_sents = len(src)

        src = self.lang_processor(src, self.src_lang, ft_model, src_nlp)
        tgt = self.lang_processor(tgt, self.tgt_lang, ft_model, tgt_nlp)

        invalid_idxs = {idx for idx, tupl in enumerate(src) if not tupl[1]}
        invalid_idxs.update({idx for idx, tupl in enumerate(tgt) if not tupl[1]})

        if invalid_idxs:
            src, tgt = self._delete_idxs(src, tgt, idxs=invalid_idxs)

        if self.max_ratio:
            src, tgt = self.check_ratio(src, tgt)

        n_final_sents = len(src)

        src_sentences = '\n'.join([s[0] for s in src]) + '\n'
        tgt_sentences = '\n'.join([s[0] for s in tgt]) + '\n'

        return src_sentences, tgt_sentences, n_init_sents, n_final_sents

    def lang_processor(self, batch, lang, ft_model, nlp=None):
        sentences = []
        docs = nlp.pipe(batch)
        for doc in docs:
            for sent in doc.sents:
                tokens = list(sent)
                n_tokens = len(tokens)
                # only count tokens that consist of alphanum chars and not only digits
                n_valid_tokens = len([t for t in tokens if t.text.isalnum() and not t.is_digit])
                len_valid = self.min_length <= n_valid_tokens <= self.max_length

                lang_label, _ = ft_model.predict(sent.text)
                lang_valid = lang_label[0].replace('__label__', '') == lang

                if self.tokenize:
                    sent = ' '.join([t.text for t in tokens])
                else:
                    sent = sent.text

                sentences.append((sent, len_valid and lang_valid, n_tokens))

        return sentences

    def writer(self):
        """ Results are not necessarily returned in order, so use prev_chunk_end
            to ensure the correct output order. """
        pfin = self.chunker.pfin
        pf_src = pfin.with_suffix(f".{self.src_lang}")
        pf_tgt = pfin.with_suffix(f".{self.tgt_lang}")

        with pf_src.open('w', encoding='utf-8') as fh_src, pf_tgt.open('w', encoding='utf-8') as fh_tgt:
            n_batch = 0
            results = {}
            prev_chunk_end = 0
            while True:
                work = self.result_queue.get()
                if work == 'done':
                    break
                chunk_start, chunk_size, src_sentences, tgt_sentences = work

                if not prev_chunk_end or prev_chunk_end == chunk_start:
                    n_batch += 1
                    logging.info(f"Processed batch {n_batch:,}...")
                    fh_src.write(src_sentences)
                    fh_tgt.write(tgt_sentences)
                    prev_chunk_end += chunk_size

                    # check if existing data in the results follows
                    # the newly added data
                    if results:
                        while True:
                            nxt = results.pop(prev_chunk_end, None)
                            if nxt:
                                n_batch += 1
                                logging.info(f"Processed batch {n_batch:,}...")
                                fh_src.write(nxt['src'])
                                fh_tgt.write(nxt['tgt'])
                                prev_chunk_end += nxt['chunk_size']
                            else:
                                break
                else:
                    results[chunk_start] = {
                        'src': src_sentences,
                        'tgt': tgt_sentences,
                        'chunk_size': chunk_size
                    }

    def reader(self):
        for chunk_tuple in self.chunker.chunkify():
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
            if src_tuple[2] / tgt_tuple[2] > self.max_ratio:
                logging.info(f"Invalid ratio for sentences: {src_tuple[0]} - {tgt_tuple[0]}")
                continue
            valid_src.append(src_tuple)
            valid_tgt.append(tgt_tuple)

        return valid_src, valid_tgt
