model_name=LLM4Time
train_epochs=100
learning_rate=0.001
llm_model=GPT2
llm_layers=6

master_port=11000
num_process=4
batch_size=62
d_model=32
d_ff=128

num_prototypes=100
type_get_prototypes='linear'

comment='LLM4Time-ETTm1-linear-100-For-Visualization'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
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
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --type_get_prototypes $type_get_prototypes \
  --num_prototypes $num_prototypes \
  --output_attention \
  --visual
