#!/usr/bin/env python3 -u
# Copyright (c) Guangsheng Bao.
# Vu Hoang edit segment function
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as path
import numpy as np
from utils import load_lines_special, save_lines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger()


def convert_to_segment(args):
    # Doc with sequence tag assigned to sentences by author
    def _segment_seqtag_origin(src, tgt, num=None):
        src = src.split('</s>')
        tgt = tgt.split('</s>')
        segs = []  # [max_tokens, max_segs, src, tgt]
        for idx, (s, t) in enumerate(zip(src, tgt)):
            if len(s) == 0 and len(t) == 0:
                continue
            assert len(s) > 0 and len(t) > 0
            s_toks = s.split()
            t_toks = t.split()
            max_toks = max(len(s_toks), len(t_toks)) + 2
            # count tokens
            if len(segs) > 0 and segs[-1][0] + max_toks < args.max_tokens \
                    and (num is None or segs[-1][1] < num):
                segs[-1][0] += max_toks
                segs[-1][1] += 1
                segs[-1][2] += ['<s> %s </s>' % ' '.join(s_toks)]
                segs[-1][3] += ['<s> %s </s>' % ' '.join(t_toks)]
            else:
                segs.append([max_toks, 1, ['<s> %s </s>' % ' '.join(s_toks)], ['<s> %s </s>' % ' '.join(t_toks)]])
        # output
        srcs = [' '.join(s) for _, _, s, _ in segs]
        tgts = [' '.join(t) for _, _, _, t in segs]
        return srcs, tgts

    # Vu Hoang: segment by tf-idf mode
    def _segment_seqtag_tfidf(src, tgt, num=None):
        src = src.split('</s>')
        tgt = tgt.split('</s>')
        tfidf_vectorizer = TfidfVectorizer()
        # Generate the tf-idf vectors for the corpus
        tfidf_matrix = tfidf_vectorizer.fit_transform(src)
        # compute and print the cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        src_data_final = []
        tgt_data_final = []
        for idx, vector in enumerate(cosine_sim):
            matrix_check = vector > args.tf_idf_score
            input_text_src = np.array(src)[np.where(matrix_check)]
            input_text_tgt = np.array(tgt)[np.where(matrix_check)]
            src_data_final.append(' '.join(input_text_src))
            tgt_data_final.append(' '.join(input_text_tgt))
        segs = []  # [max_tokens, max_segs, src, tgt]
        for idx, (s, t) in enumerate(zip(src_data_final, tgt_data_final)):
            if len(s) == 0 and len(t) == 0:
                continue
            assert len(s) > 0 and len(t) > 0
            s_toks = s.split()
            t_toks = t.split()
            max_toks = max(len(s_toks), len(t_toks)) + 2
            # count tokens
            if len(segs) > 0 and segs[-1][0] + max_toks < args.max_tokens \
                    and (num is None or segs[-1][1] < num):
                segs[-1][0] += max_toks
                segs[-1][1] += 1
                segs[-1][2] += ['<s> %s </s>' % s]
                segs[-1][3] += ['<s> %s </s>' % t]
            else:
                segs.append([max_toks, 1, ['<s> %s </s>' % s], ['<s> %s </s>' % t]])
        # output
        srcs = [' '.join(s) for _, _, s, _ in segs]
        tgts = [' '.join(t) for _, _, _, t in segs]
        return srcs, tgts

    # Vu Hoang: Doc with sequence tag assigned to sentences by multiple sentences
    def _segment_seqtag_bynum(src, tgt, num=None):
        src = src.split('</s>')
        tgt = tgt.split('</s>')
        # Convert each sentences to group 3 or 5 or 7 of sentences.
        div_src = []
        div_tgt = []
        for idx, chunk in enumerate(list(divide_chunks(src, args.divided_group_sentences))):
            div_src.append(".".join(chunk))
        for idx, chunk in enumerate(list(divide_chunks(tgt, args.divided_group_sentences))):
            div_tgt.append(".".join(chunk))
        segs = []  # [max_tokens, max_segs, src, tgt]
        for idx, (s, t) in enumerate(zip(div_src, div_tgt)):
            if len(s) == 0 and len(t) == 0:
                continue
            assert len(s) > 0 and len(t) > 0
            s_toks = s.split()
            t_toks = t.split()
            max_toks = max(len(s_toks), len(t_toks)) + 2
            # count tokens
            if len(segs) > 0 and segs[-1][0] + max_toks < args.max_tokens \
                    and (num is None or segs[-1][1] < num):
                segs[-1][0] += max_toks
                segs[-1][1] += 1
                segs[-1][2] += ['<s> %s </s>' % ' '.join(s_toks)]
                segs[-1][3] += ['<s> %s </s>' % ' '.join(t_toks)]
            else:
                segs.append([max_toks, 1, ['<s> %s </s>' % ' '.join(s_toks)], ['<s> %s </s>' % ' '.join(t_toks)]])
        # output
        srcs = [' '.join(s) for _, _, s, _ in segs]
        tgts = [' '.join(t) for _, _, _, t in segs]
        return srcs, tgts

    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    logger.info('Building segmented data: %s' % args)
    # specify the segment function
    seg_func = None
    mode_segment = args.mode_segment
    if mode_segment == 'tf_idf':
        seg_func = _segment_seqtag_tfidf
    elif mode_segment == 'number':
        seg_func = _segment_seqtag_bynum
    else:
        seg_func = _segment_seqtag_origin
    # train, valid, test
    corpuses = args.corpuses.split(',')
    for corpus in corpuses:
        source_lang_file = '%s.%s' % (corpus, args.source_lang)
        target_lang_file = '%s.%s' % (corpus, args.target_lang)
        src_lines = load_lines_special(path.join(args.datadir, source_lang_file))
        tgt_lines = load_lines_special(path.join(args.datadir, target_lang_file))
        # verify the data
        assert len(src_lines) == len(tgt_lines)
        src_sents = [len(line.split('</s>')) for line in src_lines]
        tgt_sents = [len(line.split('</s>')) for line in tgt_lines]
        assert np.all(np.array(src_sents) == np.array(tgt_sents))
        # convert
        processed = []
        src_data = []
        tgt_data = []
        for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            # check min doc length for training
            if corpus == 'train' and len(src.split()) < args.min_train_doclen:
                logger.warning('Skip too short document: corpus=train, doc=%s, sents=%s, tokens=%s'
                               % (idx, len(src.split('</s>')), len(src.split())))
                continue
            # segment the doc
            srcs, tgts = seg_func(src, tgt, args.max_sents)
            # verify doc length
            srcs_len = [len(line.split()) for line in srcs]
            if any(l > args.max_tokens for l in srcs_len):
                logger.warning('Source doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
                               % (corpus, idx, len(src.split('</s>')), max(srcs_len), args.max_tokens))
            tgts_len = [len(line.split()) for line in tgts]
            if any(l > args.max_tokens for l in tgts_len):
                logger.warning('Target doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
                               % (corpus, idx, len(tgt.split('</s>')), max(tgts_len), args.max_tokens))
            # persist
            src_data.extend(srcs)
            tgt_data.extend(tgts)
            processed.append(idx)
        # remove special token
        if args.no_special_tok:
            src_data = [line.replace('<s> ', '').replace(' </s>', '') for line in src_data]
            tgt_data = [line.replace('<s> ', '').replace(' </s>', '') for line in tgt_data]
        # save segmented language files
        logger.info('Processed %s documents of %s with a max_len of %s.' % (len(processed), corpus, args.max_tokens))
        source_lang_file = '%s.%s' % (corpus, args.source_lang)
        target_lang_file = '%s.%s' % (corpus, args.target_lang)
        source_lang_file = path.join(args.destdir, source_lang_file)
        save_lines(source_lang_file, src_data)
        logger.info('Saved %s lines into %s' % (len(src_data), source_lang_file))
        target_lang_file = path.join(args.destdir, target_lang_file)
        save_lines(target_lang_file, tgt_data)
        logger.info('Saved %s lines into %s' % (len(tgt_data), target_lang_file))


''' Generate aligned parallel text
      </s> - separator between sentences
    e.g.
      X: w1 w2 </s> w3 w4 </s>
      Y: w1 w2 w3 </s> w4 w5 </s>
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpuses", default='test,valid,train')
    parser.add_argument("--source-lang", default='en')
    parser.add_argument("--target-lang", default='de')
    parser.add_argument('--datadir', default='../exp_randinit/iwslt17.tokenized.en-de/')
    parser.add_argument("--destdir", default='../exp_randinit/iwslt17.segmented.en-de/')
    parser.add_argument("--max-sents", default=1000, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--min-train-doclen", default=-1, type=int)
    # normal / number / tf_idf
    parser.add_argument("--mode-segment", default='normal')
    parser.add_argument("--tf-idf-score", default=0.2, type=float)
    parser.add_argument("--divided-group-sentences", default=3, type=int)
    parser.add_argument('--no-special-tok', action='store_true', default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='./data_builder.log',
                        format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    os.makedirs(args.destdir, exist_ok=True)
    args.tempdir = path.join(args.destdir, 'tmp')
    os.makedirs(args.tempdir, exist_ok=True)

    convert_to_segment(args)
