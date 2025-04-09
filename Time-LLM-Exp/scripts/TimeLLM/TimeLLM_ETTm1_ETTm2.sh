model_name=TimeLLM
learning_rate=0.001
llm_layers=6
llm_model=GPT2

master_port=11000
num_process=3
batch_size=62
d_model=32
d_ff=128

num_prototypes=100
comment='TimeLLM-ETTm1_ETTm2'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTm1.csv \
  --data_path ETTm2.csv \
  --model_id ETTm1_ETTm2_512_96 \
  --model $model_name \
  --data_pretrain ETTm1 \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --llm_model $llm_model \
  --train_epochs 5 \
  --num_prototypes $num_prototypes \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTm1.csv \
  --data_path ETTm2.csv \
  --model_id ETTm1_ETTm2_512_192 \
  --model $model_name \
  --data_pretrain ETTm1 \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --llm_model $llm_model \
  --train_epochs 5 \
  --num_prototypes $num_prototypes \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTm1.csv \
  --data_path ETTm2.csv \
  --model_id ETTm1_ETTm2_512_336 \
  --model $model_name \
  --data_pretrain ETTm1 \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --llm_model $llm_model \
  --train_epochs 5 \
  --num_prototypes $num_prototypes \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTm1.csv \
  --data_path ETTm2.csv \
  --model_id ETTm1_ETTm2_512_720 \
  --model $model_name \
  --data_pretrain ETTm1 \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --llm_model $llm_model \
  --train_epochs 5 \
  --num_prototypes $num_prototypes \
  --model_comment $comment
