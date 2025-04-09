cd visualization

llm_name=GPT2
epoch=10
data=ETTh1

python visual.py --function prototype \
 --embeddings_path ../wte_pca_100.pt \
 --output_path pca \
 --llm_name ${llm_name}

python visual.py --function attention_align \
 --embeddings_path ../wte_pca_100.pt \
 --attention_path att_e${epoch}_${data}.pt \
 --output_path pca \
 --llm_name ${llm_name}

python visual.py --function aligned_tt_words \
 --embeddings_path aligned_textual_token_e${epoch}_${data}.pt \
 --selected_words_path selected_words.txt \
 --output_path pca \
 --llm_name ${llm_name}