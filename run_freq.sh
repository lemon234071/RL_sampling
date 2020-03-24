DATA_DIR="data_ost"
DATASET="freq"

if [ "$2" = "preprocess" ]; then
  python3 preprocess.py -train_src "$DATA_DIR"/src-train.txt -train_tgt "$DATA_DIR"/tgt-train.txt \
    -valid_src "$DATA_DIR"/src-valid.txt -valid_tgt "$DATA_DIR"/tgt-valid.txt \
    -train_tag_src "$DATA_DIR"/"$DATASET"/"$DATASET"-src-train.txt -train_tag_tgt "$DATA_DIR"/"$DATASET"/"$DATASET"-tgt-train.txt \
    -valid_tag_src "$DATA_DIR"/"$DATASET"/"$DATASET"-src-valid.txt -valid_tag_tgt "$DATA_DIR"/"$DATASET"/"$DATASET"-tgt-valid.txt \
    -save_data "$DATA_DIR"/"$DATASET" -share_vocab \
    -src_vocab "$DATA_DIR"/vocab.txt -tgt_vocab "$DATA_DIR"/vocab.txt
elif [ "$2" = "train" ]; then
  if [ "$3" = "seq2seq" ]; then
    # CUDA_VISIBLE_DEVICES="$1" python3 train.py -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_seq2seq -log_file ./log_dir/"$DATA_DIR"_seq2seq.log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 10000 -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
    CUDA_VISIBLE_DEVICES="$1" python3 train.py -high_num 100 \
      -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_seq2seq_"$4" \
      -log_file ./log_dir/"$DATA_DIR"_seq2seq_"$4".log -world_size 1 -gpu_ranks 0 -global_attention mlp \
      -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 50000 \
      -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 \
      -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
  elif [ "$3" = "freq" ]; then
    CUDA_VISIBLE_DEVICES="$1" python3 train.py -mask_attn -tag_gen "$4" -high_num 100 -generators high:104,low:49900 \
      -itoj "$DATA_DIR"/"$DATASET"/itoj.json -data "$DATA_DIR"/"$DATASET" \
      -save_model checkpoint/"$DATA_DIR"_freq_"$4"_"$5" -log_file ./log_dir/"$DATA_DIR"_freq_"$4"_"$5".log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 10000 -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -pos_vec_size 128 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
  elif [ "$3" = "tri" ]; then
    CUDA_VISIBLE_DEVICES="$1" python3 train.py -generators stop:175,vn:35985,ord:13844 -mask_attn -tag_gen "$4" -itoj "$DATA_DIR"/tri_itoj.json -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_"$DATASET"_"$4"_"$5" -log_file ./log_dir/"$DATA_DIR"_"$DATASET"_"$4"_"$5".log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 10000 -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -pos_vec_size 128 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
  else
    echo "Input arg 3 Is Error: $3."
  fi
elif [ "$2" = "infer" ]; then
  python3 translate.py -gpu "$1" -model "$3" -sample_method "$4" -tag_mask "$DATA_DIR"/"$DATASET"/"$DATASET"_mask.json \
    -output result/"$DATA_DIR"_"$DATASET"_"$4"_"$5".txt -beam 1 -batch_size 128 \
    -src "$DATA_DIR"/src-test.txt -max_length 30 -tag_src "$DATA_DIR"/"$DATASET"/"$DATASET"-src-test.txt
elif [ "$2" = "rl_train" ]; then
  CUDA_VISIBLE_DEVICES="$1" python3 rl_train.py -gpu 0 -model "$3" -sample_method "$4" \
    -generators high:104,low:49900 -tag_mask "$DATA_DIR"/"$DATASET"/"$DATASET"_mask.json -rl_samples 4 -epochs 3 \
    -batch_size 1024 -report_every 10 -valid_steps 50 -save_checkpoint_steps 100 -random_steps 400 \
    -learning_rate 0.001 -learning_rate_decay 0.8 -start_decay_steps 5000 -decay_steps 200 \
    -output result/rl_"$DATA_DIR"_"$DATASET"_"$4"_"$5".txt -save_model rl_checkpoint/mlp_"$4"_"$5" \
    -data "$DATA_DIR"/"$DATASET" -src "$DATA_DIR"/src-train.txt -tgt "$DATA_DIR"/tgt-train.txt \
    -valid_src "$DATA_DIR"/src-valid.txt -valid_tgt "$DATA_DIR"/tgt-valid.txt \
    -tag_src "$DATA_DIR"/"$DATASET"/"$DATASET"-src-train.txt -tag_tgt "$DATA_DIR"/"$DATASET"/"$DATASET"-tgt-train.txt \
    -valid_tag_src "$DATA_DIR"/"$DATASET"/"$DATASET"-src-valid.txt -valid_tag_tgt "$DATA_DIR"/"$DATASET"/"$DATASET"-tgt-valid.txt \
    -optim adam -beam 1 -max_length 30
elif [ "$2" = "rl_infer" ]; then
  CUDA_VISIBLE_DEVICES="$1" python3 rl_train.py -infer -train_from rl_checkpoint/mlp_freq_v2_step_900.pt
else
  echo "Input arg 2 Is Error: $2."
fi
# test
# python3 train.py -data ./data/data -save_model checkpoint/test -world_size 1 -gpu_ranks 0 -global_attention mlp -batch_size 16 -log_file ./log_dir/test.log -word_vec_size 512 -warmup_steps 5000 -train_steps 200000 -optim adam -adam_beta2 0.998 -param_init_glorot -decay_method noam -max_generator_batches 128 -rnn_size 512 -rnn_type GRU -report_every 500 -pos_vec_size 100 -tag_gen
