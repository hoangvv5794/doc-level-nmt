import argparse
import logging
import os
import os.path as path

from sentence_transformers import SentenceTransformer, util

from utils import read_file, save_lines

model = SentenceTransformer("all-MiniLM-L6-v2")
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


def convert_to_segment(args):
    min_size_chunk = args.lower_size_chunk
    max_size_chunk = args.upper_size_chunk
    threshold_similarity = args.threshold_similarity
    logger.info('Building segmented data: %s' % args)
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
            matrix_cosine_similarity = util.cos_sim(embeddings, embeddings)
            src_data_out = []
            tgt_data_out = []
            src_current_chunk = []
            tgt_current_chunk = []
            size_of_doc = len(src_lines_split)
            for index, (src_sentence, tgt_sentence) in enumerate(
                    zip(src_lines_split, tgt_lines_split)):
                # process with each sentence from source and target
                current_size = len(src_current_chunk)
                # size of current chunk smaller than min_size => append sentence to current chunk
                # size of current chunk larger than max_size => create new chunk with sentence + append chunk to result
                if current_size < min_size_chunk:
                    src_current_chunk.append(src_sentence)
                    tgt_current_chunk.append(tgt_sentence)
                elif current_size >= max_size_chunk:
                    src_data_out.append(src_current_chunk)
                    tgt_data_out.append(tgt_current_chunk)
                    src_current_chunk = []
                    tgt_current_chunk = []
                    src_current_chunk.append(src_sentence)
                    tgt_current_chunk.append(tgt_sentence)
                else:
                    # define sentence belong current_chunk or next_chunk
                    # if similarity of current > threshold (0.6) => append sentence to current chunk
                    average_similarity_current = 0
                    for i in range(1, current_size + 1):
                        average_similarity_current += matrix_cosine_similarity[index][index - i]
                    average_similarity_current = average_similarity_current / current_size
                    if index >= size_of_doc - 1:
                        similarity_next = threshold_similarity
                    else:
                        similarity_next = matrix_cosine_similarity[index][index + 1]
                    if (average_similarity_current > similarity_next) or average_similarity_current > threshold_similarity:
                        # current sentence belong current chunk => append to current chunk
                        src_current_chunk.append(src_sentence)
                        tgt_current_chunk.append(tgt_sentence)
                    else:
                        # current sentence belong next chunk => create new chunk
                        src_data_out.append(src_current_chunk)
                        tgt_data_out.append(tgt_current_chunk)
                        src_current_chunk = []
                        tgt_current_chunk = []
                        src_current_chunk.append(src_sentence)
                        tgt_current_chunk.append(tgt_sentence)
            if len(src_current_chunk) > 0:
                src_data_out.append(src_current_chunk)
                tgt_data_out.append(tgt_current_chunk)
            src_data_final = []
            tgt_data_final = []
            for src_data_out_sentence in src_data_out:
                src_data_final.append(' %s </s>' % ' '.join(src_data_out_sentence))
            for tgt_data_out_sentence in tgt_data_out:
                tgt_data_final.append(' %s </s>' % ' '.join(tgt_data_out_sentence))
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
    parser.add_argument("--corpora", default='test')
    parser.add_argument("--dataset", default='iwslt17/')
    parser.add_argument("--source-lang", default='en')
    parser.add_argument("--target-lang", default='de')
    parser.add_argument('--datadir', default='../raw_data/')
    parser.add_argument("--destdir", default='../exp_randinit/')
    parser.add_argument("--max-sents", default=1000, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--min-train-doclen", default=-1, type=int)
    # max and min size chunk of sentence
    parser.add_argument("--lower-size-chunk", default=1, type=int)
    parser.add_argument("--upper-size-chunk", default=5, type=int)
    parser.add_argument("--threshold-similarity", default=0.6, type=float)

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
