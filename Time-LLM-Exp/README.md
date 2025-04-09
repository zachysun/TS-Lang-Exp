# Time-LLM-Experiments

**Time-LLM: Time Series Forecasting by Reprogramming Large Language Models** [[PDF]](https://arxiv.org/abs/2310.01728) [[Official Code](https://github.com/KimMeen/Time-LLM)]

```latex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Models

- Base

- Base_Prompt

- Base_Prototype

- TimeLLM

## Run Ablation Experiments

### Part 1

#### Long-term Forecasting (Q1)

```bash
bash ./scripts/Base/Base_ETTh1.sh
bash ./scripts/Base_Prompt/Base_Prompt_ETTh1.sh
bash ./scripts/Base_Prototype/Base_Prototype_ETTh1.sh
bash ./scripts/TimeLLM/TimeLLM_ETTh1.sh
...
```

#### Few-shot Forecasting (Q2)

change argument "percent=100" to "percent=10"

#### Zero-shot Forecasting (Q3)

```bash
bash ./scripts/Base/Base_ETTh1_ETTh2.sh
bash ./scripts/Base_Prompt/Base_Prompt_ETTh1_ETTh2.sh
bash ./scripts/Base_Prototype/Base_Prototype_ETTh1_ETTh2.sh
bash ./scripts/TimeLLM/TimeLLM_ETTh1_ETTh2.sh
...
```

### Part 2

#### Text Prototypes Extraction Method (Q4)

```bash
bash ./scripts/TimeLLM/TimeLLM_ETTh1.sh
bash ./scripts/LLM4Time/Ablation_pmethod/LLM4Time_ETTh1_kmeans.sh
bash ./scripts/LLM4Time/Ablation_pmethod/LLM4Time_ETTh1_pca.sh
bash ./scripts/LLM4Time/Ablation_pmethod/LLM4Time_ETTh1_random.sh
bash ./scripts/LLM4Time/Ablation_pmethod/LLM4Time_ETTh1_similarity.sh
bash ./scripts/LLM4Time/Ablation_pmethod/LLM4Time_ETTh1_text.sh
```

#### Number of Text Prototypes (Q5)

```bash
bash ./scripts/LLM4Time/Ablation_pnums/LLM4Time_ETTh1_linear_pnums10.sh
bash ./scripts/LLM4Time/Ablation_pnums/LLM4Time_ETTh1_linear_pnums20.sh
bash ./scripts/LLM4Time/Ablation_pnums/LLM4Time_ETTh1_linear_pnums50.sh
...
```

#### Hidden Dimension of Text Prototypes (Q6)

```bash
bash ./scripts/LLM4Time/Ablation_pdims/LLM4Time_ETTh1_linear_pdim16.sh
bash ./scripts/LLM4Time/Ablation_pdims/LLM4Time_ETTh1_linear_pdim32.sh
bash ./scripts/LLM4Time/Ablation_pdims/LLM4Time_ETTh1_linear_pdim64.sh
...
```

### Part 3

#### Random Replaced Text Prompts and Text Prototypes (Q7)

```bash
bash ./scripts/TimeLLM/TimeLLM_ETTh1_visual.sh
bash ./scripts/TimeLLM/TimeLLM_replace.sh
```


## Run Visualization

```bash
bash ./scripts/visualize.sh
```

