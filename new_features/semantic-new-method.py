import argparse
import logging
import os
import os.path as path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import read_file, save_lines, calculate_cosine_distances_dictionary

model = SentenceTransformer("all-mpnet-base-v2")
logger = logging.getLogger()


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


def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['source'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['source']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['source']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_source'] = combined_sentence

    return sentences


def convert_to_segment(args):
    # train, valid, test
    dataset = args.dataset
    corpora = args.corpora.split(',')
    for corpus in corpora:
        if corpus == 'valid':
            corpus = 'dev'
        source_lang_file = 'concatenated_en2de_%s_%s.txt' % (corpus, args.source_lang)
        target_lang_file = 'concatenated_en2de_%s_%s.txt' % (corpus, args.target_lang)
        src_raw = read_file(path.join(args.datadir, dataset, source_lang_file))
        tgt_raw = read_file(path.join(args.datadir, dataset, target_lang_file))
        src_doc_split = src_raw.split('<d>')
        tgt_doc_split = tgt_raw.split('<d>')
        src_doc_split = list(filter(None, src_doc_split))
        tgt_doc_split = list(filter(None, tgt_doc_split))
        src_data = []
        tgt_data = []

        for idx, (src_doc, tgt_doc) in enumerate(zip(src_doc_split, tgt_doc_split)):
            # process for each source_doc and target_doc
            # split document to each sentence
            src_lines_split = src_doc.split('\r\n')
            tgt_lines_split = tgt_doc.split('\r\n')
            src_lines_split = list(filter(None, src_lines_split))
            tgt_lines_split = list(filter(None, tgt_lines_split))
            # push each sentence to BERT Model and vectorize each sentence
            embeddings = model.encode(src_lines_split, convert_to_tensor=True)
            data_set = [{'source': x, 'index': i, 'target': y} for i, (x, y) in
                        enumerate(zip(src_lines_split, tgt_lines_split))]
            data_set = combine_sentences(data_set)
            for i, sentence in enumerate(data_set):
                sentence['combined_sentence_embedding'] = embeddings[i]
            distances, data_set = calculate_cosine_distances_dictionary(data_set)

            breakpoint_percentile_threshold = 80
            breakpoint_distance_threshold = np.percentile(distances,
                                                          breakpoint_percentile_threshold)
            indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
            start_index = 0

            # Create a list to hold the grouped sentences
            src_data_final = []
            tgt_data_final = []
            # Iterate through the breakpoints to slice the sentences
            for index in indices_above_thresh:
                # The end index is the current breakpoint
                end_index = index
                # Slice the sentence_dicts from the current start index to the end index
                group = data_set[start_index:end_index + 1]
                combined_src = ' '.join([d['source'] for d in group])
                combined_tgt = ' '.join([d['target'] for d in group])
                src_data_final.append('%s </s>' % combined_src)
                tgt_data_final.append('%s </s>' % combined_tgt)
                # Update the start index for the next group
                start_index = index + 1

            # The last group, if any sentences remain
            if start_index < len(data_set):
                combined_src = ' '.join([d['source'] for d in data_set[start_index:]])
                combined_tgt = ' '.join([d['target'] for d in data_set[start_index:]])
                src_data_final.append('%s </s>' % combined_src)
                tgt_data_final.append('%s </s>' % combined_tgt)

            num = args.max_sents
            srcs, tgts = _segment_seqtag_origin(' '.join(src_data_final), ' '.join(tgt_data_final), num)
            src_data.extend(srcs)
            tgt_data.extend(tgts)
        # write out to file
        if corpus == 'dev':
            corpus = 'valid'
        source_lang_file = '%s.%s' % (corpus, args.source_lang)
        target_lang_file = '%s.%s' % (corpus, args.target_lang)
        source_lang_file = path.join(args.destdir, source_lang_file)
        save_lines(source_lang_file, src_data)
        logger.info('Saved %s lines into %s' % (len(src_data), source_lang_file))
        target_lang_file = path.join(args.destdir, target_lang_file)
        save_lines(target_lang_file, tgt_data)
        logger.info('Saved %s lines into %s' % (len(tgt_data), target_lang_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", default='test,valid')
    parser.add_argument("--dataset", default='iwslt17/')
    parser.add_argument("--source-lang", default='en')
    parser.add_argument("--target-lang", default='de')
    parser.add_argument('--datadir', default='../raw_data/')
    parser.add_argument("--destdir", default='../exp_randinit/')
    parser.add_argument("--max-sents", default=1000, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--min-train-doclen", default=-1, type=int)

    parser.add_argument("--mode-segment", default='semantic')
    parser.add_argument('--no-special-tok', action='store_true', default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='../exp_gtrans/data_builder.log',
                        format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    os.makedirs(args.destdir, exist_ok=True)
    args.tempdir = path.join(args.destdir, 'tmp')
    os.makedirs(args.tempdir, exist_ok=True)

    convert_to_segment(args)
