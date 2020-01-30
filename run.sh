if [[ $1 == 'preprocess' ]]; then
  python3 preprocess.py -train_src data_reddit_small/src-train.txt -train_tgt data_reddit_small/tgt-train.txt -valid_src data_reddit_small/src-valid.txt -valid_tgt data_reddit_small/tgt-valid.txt -save_data data_reddit_small/freq -train_pos_src data_reddit_small/freq-src-train.txt -train_pos_tgt data_reddit_small/freq-tgt-train.txt -valid_pos_src data_reddit_small/freq-src-valid.txt -valid_pos_tgt data_reddit_small/freq-tgt-valid.txt -share_vocab
elif [[ $1 == 'train' ]]; then
  if [[ $2 == 'seq2seq' ]]; then
    python3 train.py -data ./data_reddit_small/freq -save_model checkpoint/reddit_small_seq2seq -world_size 1 -gpu_ranks 0 -global_attention mlp -log_file ./log_dir/reddit_small_seq2seq.log -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -decay_method noam -rnn_type GRU -warmup_steps 5000 -train_steps 200000 -batch_size 128 -max_generator_batches 128 -report_every 500 -valid_steps 1000 -save_checkpoint_steps 1000
  elif [[ $2 == 'freq' ]]; then
    python3 train.py -data ./data_reddit_small/freq -save_model checkpoint/reddit_small_freq -world_size 1 -gpu_ranks 0 -global_attention mlp -log_file ./log_dir/reddit_small_freq.log -word_vec_size 256 -rnn_size 512 -optim adam -param_init_glorot -decay_method noam -rnn_type GRU -warmup_steps 5000 -train_steps 200000 -batch_size 128 -max_generator_batches 128 -report_every 500 -pos_vec_size 128 -valid_steps 1000 -save_checkpoint_steps 1000 -pos_gen
  else
    echo "Input arg 2 Is Error."
  fi
elif [[ $1 == 'infer' ]]; then
  python3 translate.py
else
  echo "Input arg 1 Is Error."
fi
# test
# python3 train.py -data ./data/data -save_model checkpoint/test -world_size 1 -gpu_ranks 0 -global_attention mlp -batch_size 16 -log_file ./log_dir/test.log -word_vec_size 512 -warmup_steps 5000 -train_steps 200000 -optim adam -adam_beta2 0.998 -param_init_glorot -decay_method noam -max_generator_batches 128 -rnn_size 512 -rnn_type GRU -report_every 500 -pos_vec_size 100 -pos_gen
