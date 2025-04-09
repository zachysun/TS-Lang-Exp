model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llm_model=LLAMA
llm_layers=8

master_port=15000
num_process=2
batch_size=8
d_model=32
d_ff=128
percent=100

num_prototypes=1000


for replace_prompts_rate in 0 10 40 70 100
do
comment=TimeLLM-ETTh2-replace_prompts_rate-$replace_prompts_rate

  for pred_len in 96 192 336 720
  do

    accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_512_${pred_len} \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len $pred_len \
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
      --num_prototypes $num_prototypes \
      --percent $percent \
      --replace_prompts_rate $replace_prompts_rate

  done
done

for replace_words_rate in 0 10 40 70 100
do
comment=TimeLLM-ETTh2-replace_words_rate-$replace_words_rate

  for pred_len in 96 192 336 720
  do

    accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_512_${pred_len} \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len $pred_len \
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
      --num_prototypes $num_prototypes \
      --percent $percent \
      --replace_words_rate $replace_words_rate

  done
done