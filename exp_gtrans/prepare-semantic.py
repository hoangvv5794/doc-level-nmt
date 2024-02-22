import argparse
import logging
import os
import os.path as path

from sentence_transformers import SentenceTransformer, util

from utils import read_file

model = SentenceTransformer("all-MiniLM-L6-v2")
logger = logging.getLogger()


def convert_to_segment(args):
    min_size_chunk = args.lower_size_chunk
    max_size_chunk = args.upper_size_chunk
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
        for idx, (src_doc, tgt_doc) in enumerate(zip(src_doc_split, tgt_doc_split)):
            # process for each source_doc and target_doc
            # split document to each sentence
            src_lines_split = src_doc.split('\r\n')
            tgt_lines_split = tgt_doc.split('\r\n')
            # push each sentence to BERT Model and vectorize each sentence
            embeddings = model.encode(src_lines_split, convert_to_tensor=True)
            # Print the embeddings
            src_data_out = []
            tgt_data_out = []
            src_current_chunk = []
            tgt_current_chunk = []
            for idx, (src_sentence, tgt_sentence) in enumerate(
                    zip(src_lines_split, tgt_lines_split)):
                # compare and merge sentence to groups
                # split each group with delimiter <s>
                current_size = len(src_current_chunk)
                if current_size < min_size_chunk:
                    src_current_chunk.append(src_sentence)
                    tgt_current_chunk.append(tgt_sentence)
                elif current_size >= max_size_chunk:
                    src_data_out.append(src_current_chunk)
                    tgt_data_out.append(tgt_current_chunk)
                    src_current_chunk = []
                    tgt_current_chunk = []
                else:
                    # define current_sentence belong current_chunk or next_chunk
                    # embedding of current_chunk
                    current_chunk_embed = embeddings[idx - current_size:idx - 1]
                    next_chunk_embed = embeddings[idx + 1]
                    # find out current_embedding belong to current chunk or next chunk
                    cosine_matrix_current_chunk = util.cos_sim(embeddings[idx], current_chunk_embed)
                    cosine_matrix_next_chunk = util.cos_sim(embeddings[idx], next_chunk_embed)
                    print(cosine_matrix_next_chunk)
                    print(cosine_matrix_current_chunk)
            # write out to file

    # split by document

    # convert
    # processed = []
    # src_data = []
    # tgt_data = []
    # for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
    #     # check min doc length for training
    #     if corpus == 'train' and len(src.split()) < args.min_train_doclen:
    #         logger.warning('Skip too short document: corpus=train, doc=%s, sents=%s, tokens=%s'
    #                        % (idx, len(src.split('</s>')), len(src.split())))
    #         continue
    #     # segment the doc
    #     srcs, tgts = seg_func(src, tgt, args.max_sents)
    #     # verify doc length
    #     srcs_len = [len(line.split()) for line in srcs]
    #     if any(l > args.max_tokens for l in srcs_len):
    #         logger.warning('Source doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
    #                        % (corpus, idx, len(src.split('</s>')), max(srcs_len), args.max_tokens))
    #     tgts_len = [len(line.split()) for line in tgts]
    #     if any(l > args.max_tokens for l in tgts_len):
    #         logger.warning('Target doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
    #                        % (corpus, idx, len(tgt.split('</s>')), max(tgts_len), args.max_tokens))
    #     # persist
    #     src_data.extend(srcs)
    #     tgt_data.extend(tgts)
    #     processed.append(idx)
    # # remove special token
    # if args.no_special_tok:
    #     src_data = [line.replace('<s> ', '').replace(' </s>', '') for line in src_data]
    #     tgt_data = [line.replace('<s> ', '').replace(' </s>', '') for line in tgt_data]
    # # save segmented language files
    # logger.info('Processed %s documents of %s with a max_len of %s.' % (len(processed), corpus, args.max_tokens))
    # source_lang_file = '%s.%s' % (corpus, args.source_lang)
    # target_lang_file = '%s.%s' % (corpus, args.target_lang)
    # source_lang_file = path.join(args.destdir, source_lang_file)
    # save_lines(source_lang_file, src_data)
    # logger.info('Saved %s lines into %s' % (len(src_data), source_lang_file))
    # target_lang_file = path.join(args.destdir, target_lang_file)
    # save_lines(target_lang_file, tgt_data)
    # logger.info('Saved %s lines into %s' % (len(tgt_data), target_lang_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", default='test,valid,train')
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
