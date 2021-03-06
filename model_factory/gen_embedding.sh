#!/usr/bin/env bash
PATH=$PATH:/home/$(whoami)/anaconda3/condabin:/home/$(whoami)/anaconda3/bin
echo $PATH
source activate ml-dev
cur_dir=$(pwd)
data_folder=$cur_dir/data/
model_name=uncased_L-12_H-768_A-12
BERT_BASE_DIR=$cur_dir/models/$model_name
python $cur_dir/model_factory/embeddings/bert-master/extract_features.py \
  --input_file=$data_folder/small_input.txt \
  --output_file=$data_folder/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=32