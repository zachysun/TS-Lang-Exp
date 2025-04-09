cd visualization

type_get_prototype=linear
llm_name=GPT2
epoch=10
patch_len=16
stride=8
dataset=ETTh1
num_prototypes=100

for function in prototype attention_align attention_llm aligned_ts aligned_ts_words prototype_words patches_semantic_ts
do
  if [ "$function" == "prototype" ]
  then
    python visual.py --function ${function} \
     --embeddings_path ./${type_get_prototype}/ProEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "attention_align" ]
  then
    python visual.py --function ${function} \
     --embeddings_path ./${type_get_prototype}/ProEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --attention_path ./${type_get_prototype}/Att_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "attention_llm" ]
  then
    python visual.py --function ${function} \
     --attention_path ./${type_get_prototype}/LLMAtt_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --prompts_id_path ./${type_get_prototype}/promptsId_pl${patch_len}_st${stride}_${dataset}.txt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "aligned_ts" ]
  then
    python visual.py --function ${function} \
     --embeddings_path ./${type_get_prototype}/RepEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "aligned_ts_words" ]
  then
    python visual.py --function ${function} \
     --embeddings_path ./${type_get_prototype}/RepEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --selected_words_path selected_words.txt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "prototype_words" ]
  then
    python visual.py --function ${function} \
     --embeddings_path ./${type_get_prototype}/ProEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --selected_words_path selected_words.txt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  elif [ "$function" == "patches_semantic_ts" ]
  then
    python visual.py --function ${function} \
     --patches_path ./${type_get_prototype}/patches_pl${patch_len}_st${stride}_${dataset}.txt \
     --embeddings_path ./${type_get_prototype}/RepEmb_e${epoch}_pd768_pm_${type_get_prototype}_pn${num_prototypes}_pl${patch_len}_st${stride}_${dataset}.pt \
     --output_path ${type_get_prototype} \
     --llm_name ${llm_name}
  fi
done