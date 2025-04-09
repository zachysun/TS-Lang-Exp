# CALF - Experiments

**CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning** [[PDF](https://arxiv.org/abs/2403.07300)] [[Official Code](https://github.com/Hank0626/CALF)]

```latex
@article{liu2024taming,
      title={CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning}, 
      author={Liu, Peiyuan and Guo, Hang and Dai, Tao and Li, Naiqi and Bao, Jigang and Ren, Xudong and Jiang, Yong and Xia, Shu-Tao},
      journal={arXiv preprint arXiv:2403.07300},
      year={2024},
      arxiv={2403.07300}
}
```

## Models

- CALF_Temporal
- CALF

## Run Ablation Experiments

### Part 1

#### Long-term Forecasting (Q1)

```bash
bash ./scripts/long_term_forecasting/ETTh1_Base.sh
bash ./scripts/long_term_forecasting/electricity_base.sh
...
```

## Run Visualization

```bash
bash ./scripts/long_term_forecasting/ETTh1_visual.sh
bash ./scripts/visualize.sh
```

