#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
# e.g.
# Vu Hoang edit: bash prepare-randinit.sh iwslt17 exp_test semantic

data=$1
exp_path=$2
mode_segment=$3
input=doc
code=bpe

slang=en
tlang=de

# segment mode: semantic/normal/number/tf-idf

  # Semantic mode: 03 technique: normal / optimize / new_method
  # if new_method mode: define threshold breakpoint
  # default: distances > 80% => create new chunks
  # if normal mode: define max and min size chunk of sentence
  # if optimize mode: define threshold

  # Normal mode: default by paper

  # Number mode: grouping sentences by rules: 3/5/7 sentences each group
  # define number sentence each group: --divided-group-sentences 3

  # TF-IDF mode: split by TF-IDF
  # define threshold tf-idf: --tf-idf-score 0.2

if [ $mode_segment == "semantic" ]; then
  seg_path=$exp_path/$data.segmented.$slang-$tlang
  echo `date`, seg_path: seg_path, data: $data, input: $input, code: $code, slang: $slang, tlang: $tlang
  if [ $input == "doc" ]; then
    python3 -m exp_gtrans.prepare-semantic --dataset data --datadir raw_data/ --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000 --mode-semantic new_method --threshold-breakpoint 80
  elif [ $input == "sent" ]; then
    python3 -m exp_gtrans.prepare-semantic --dataset data --datadir raw_data/ --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1
  fi
elif [ $mode_segment == "normal" ]; then
  echo `date`, exp_path: $exp_path, data: $data, input: $input, code: $code, slang: $slang, tlang: $tlang
  tok_path=$exp_path/$data.tokenized.$slang-$tlang
  seg_path=$exp_path/$data.segmented.$slang-$tlang
  bin_path=$exp_path/$data.binarized.$slang-$tlang

  echo `date`, Prepraring data...

  # tokenize and sub-word
  bash exp_gtrans/prepare-bpe.sh raw_data/$data $tok_path

  # data builder
  if [ $input == "doc" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000
  elif [ $input == "sent" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1
  fi
elif [ $mode_segment == "tf_idf" ]; then
  echo `date`, exp_path: $exp_path, data: $data, input: $input, code: $code, slang: $slang, tlang: $tlang
  tok_path=$exp_path/$data.tokenized.$slang-$tlang
  seg_path=$exp_path/$data.segmented.$slang-$tlang
  bin_path=$exp_path/$data.binarized.$slang-$tlang

  echo `date`, Prepraring data...

  # tokenize and sub-word
  bash exp_gtrans/prepare-bpe.sh raw_data/$data $tok_path

  # data builder
  if [ $input == "doc" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000 --mode-segment $mode_segment --tf-idf-score 0.2
  elif [ $input == "sent" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1
  fi
elif [ $mode_segment == "number" ]; then
  echo `date`, exp_path: $exp_path, data: $data, input: $input, code: $code, slang: $slang, tlang: $tlang
  tok_path=$exp_path/$data.tokenized.$slang-$tlang
  seg_path=$exp_path/$data.segmented.$slang-$tlang
  bin_path=$exp_path/$data.binarized.$slang-$tlang

  echo `date`, Prepraring data...

  # tokenize and sub-word
  bash exp_gtrans/prepare-bpe.sh raw_data/$data $tok_path

  # data builder
  if [ $input == "doc" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1000 --divided-group-sentences 3
  elif [ $input == "sent" ]; then
    python3 -m exp_gtrans.data_builder --datadir $tok_path --destdir $seg_path/ --source-lang $slang --target-lang $tlang --max-tokens 512 --max-sents 1 --divided-group-sentences 3
  fi
fi


# Preprocess/binarize the data
python3 -m fairseq_cli.preprocess --task translation_doc --source-lang $slang --target-lang $tlang \
       --trainpref $seg_path/train --validpref $seg_path/valid --testpref $seg_path/test --destdir $bin_path \
       --joined-dictionary --workers 8
