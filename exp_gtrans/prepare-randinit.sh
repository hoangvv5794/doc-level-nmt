#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
# e.g.
# bash prepare-randinit.sh iwslt17 exp_test

data=$1
exp_path=$2
input=doc
code=bpe

slang=en
tlang=de

echo `date`, exp_path: $exp_path, data: $data, input: $input, code: $code, slang: $slang, tlang: $tlang
tok_path=$exp_path/$data.tokenized.$slang-$tlang
seg_path=$exp_path/$data.segmented.$slang-$tlang
bin_path=$exp_path/$data.binarized.$slang-$tlang

echo `date`, Prepraring data...

# tokenize and sub-word
bash exp_gtrans/prepare-bpe.sh raw_data/$data $tok_path

# data builder
if [ $input == "doc" ]; then
  python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000 --mode-segment tf_idf --tf-idf-score 0.2
elif [ $input == "sent" ]; then
  python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1
fi

# Preprocess/binarize the data
python3 -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
       --trainpref $seg_path/train --validpref $seg_path/valid --testpref $seg_path/test --destdir $bin_path \
       --joined-dictionary --workers 8
