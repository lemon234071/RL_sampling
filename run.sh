# DATA_DIR="data_freq0.001"
DATA_DIR="data_reddit_small"
DATASET="freq"

if [ "$2" = "preprocess" ]; then
  python3 preprocess.py -train_src "$DATA_DIR"/src-train.txt -train_tgt "$DATA_DIR"/tgt-train.txt -valid_src "$DATA_DIR"/src-valid.txt -valid_tgt "$DATA_DIR"/tgt-valid.txt -save_data "$DATA_DIR"/"$DATASET" -train_pos_src "$DATA_DIR"/"$DATASET"-src-train.txt -train_pos_tgt "$DATA_DIR"/"$DATASET"-tgt-train.txt -valid_pos_src "$DATA_DIR"/"$DATASET"-src-valid.txt -valid_pos_tgt "$DATA_DIR"/"$DATASET"-tgt-valid.txt -share_vocab -src_vocab "$DATA_DIR"/vocab.txt -tgt_vocab "$DATA_DIR"/vocab.txt
elif [ "$2" = "train" ]; then
  if [ "$3" = "seq2seq" ]; then
    # CUDA_VISIBLE_DEVICES="$1" python3 train.py -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_seq2seq -log_file ./log_dir/"$DATA_DIR"_seq2seq.log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 10000 -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
    CUDA_VISIBLE_DEVICES="$1" python3 train.py -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_seq2seq_"$4" -log_file ./log_dir/"$DATA_DIR"_seq2seq_"$4".log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 50000 -learning_rate 0.001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
  elif [ "$3" = "freq" ]; then
    CUDA_VISIBLE_DEVICES="$1" python3 train.py -train_from checkpoint/data_reddit_small_freq_multi_mask_attn_step_50000.pt -t_gen -mask_attn -tag_gen "$4" -data "$DATA_DIR"/"$DATASET" -save_model checkpoint/"$DATA_DIR"_freq_"$4"_"$5" -log_file ./log_dir/"$DATA_DIR"_freq_"$4"_"$5".log -world_size 1 -gpu_ranks 0 -global_attention mlp -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -rnn_type GRU -start_decay_steps 10000 -learning_rate 0.0001 -train_steps 50000 -batch_size 128 -max_generator_batches 128 -report_every 500 -pos_vec_size 128 -valid_steps 1000 -save_checkpoint_steps 5000 -statistic
  else
    echo "Input arg 3 Is Error: $3."
  fi
elif [ "$2" = "infer" ]; then
  python3 translate.py -gpu "$1" -model "$3" -sample_method "$4" -output result/"$DATA_DIR"_"$DATASET"_"$4".txt -beam 1 -batch_size 128 -src "$DATA_DIR"/src-test.txt -max_length 30 -pos_src "$DATA_DIR"/"$DATASET"-src-test.txt
elif [ "$2" = "rl_train" ]; then
  python3 rl_train.py -sample_method "$4" -rl_samples 4 -learning_rate 0.001 -learning_rate_decay 0.8 -start_decay_steps 1000 -decay_steps 1000 -train_steps 5000 -valid_steps 500 -save_model rl_checkpoint/mlp_"$4"_"$5" -valid_src "$DATA_DIR"/src-valid.txt -valid_tgt "$DATA_DIR"/src-valid.txt -optim adam -gpu "$1" -model "$3" -output result/rl_"$DATA_DIR"_"$DATASET"_"$4"_"$5".txt -beam 1 -batch_size 128 -src "$DATA_DIR"/src-train.txt -max_length 30
elif [ "$2" = "rl_infer" ]; then
  python3 rl_train.py -infer
else
  echo "Input arg 2 Is Error: $2."
fi
# test
# python3 train.py -data ./data/data -save_model checkpoint/test -world_size 1 -gpu_ranks 0 -global_attention mlp -batch_size 16 -log_file ./log_dir/test.log -word_vec_size 512 -warmup_steps 5000 -train_steps 200000 -optim adam -adam_beta2 0.998 -param_init_glorot -decay_method noam -max_generator_batches 128 -rnn_size 512 -rnn_type GRU -report_every 500 -pos_vec_size 100 -tag_gen
