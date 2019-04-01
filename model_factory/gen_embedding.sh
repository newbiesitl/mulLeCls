cur_dir=$(pwd)
data_folder=$cur_dir/../data/
ls $data_folder

model_name=uncased_L-12_H-768_A-12
BERT_BASE_DIR=$cur_dir/../models/$model_name

python $cur_dir/embeddings/bert-master/extract_features.py \
  --input_file=$data_folder/input.txt \
  --output_file=$data_folder/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --layers=-1,-2,-3,-4 \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=8