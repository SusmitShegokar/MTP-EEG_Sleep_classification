# EEG Sleep Stage Classification

Code from my M.Tech thesis at IIT Kharagpur on automatic sleep staging from EEG, with a focus on generalising across subjects and age groups.

Related publications from this line of work:

- *XSleepFormer: A Compact CNN-Transformer for EEG Sleep Stage Classification with Cross-Subject and Cross-Age Generalization*, PReMI 2025. [Springer](https://link.springer.com/chapter/10.1007/978-3-032-18480-1_18)
- *NeuroWhisper: Evaluating speech-derived convolution-attention encoders for sleep staging from electroencephalogram signals*, Computers and Electrical Engineering, 2026. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045790626004702)

## What is here

| Folder | Contents |
|---|---|
| `FINAL_run/` | End-to-end pipeline: `main.py` runs `dataloader.py`, then trains (`model_runner.py`) and evaluates (`evaluator.py`) every model in `modelpy/` |
| `FINAL_run/ipynbRun/` | The same pipeline as a notebook, with saved outputs (metrics, confusion matrices, ROC and PR curves) |
| `Usleep/` | Earlier U-Sleep experiments (notebooks) |
| `residual on Old data/` | Earlier notebooks for the residual and CNN-Transformer-MLP models |
| `with_sub/` | Variant of the pipeline with additional experiments |

## Models compared

| Model file | Architecture |
|---|---|
| `residual` | Squeeze-and-excitation residual 1D CNN, bidirectional GRU and multi-head attention |
| `cnn-trans-mlp` | 1D CNN front end, Transformer encoder and MLP classifier |
| `winner` | Two-branch multi-scale 1D CNN |
| `USleep` | U-Sleep (from Braindecode) |
| `SleepStagerChambon2018`, `SleepStagerBlanco2020` | Braindecode baselines |

Training uses AdamW with a one-cycle learning-rate schedule and mixup augmentation, followed by fine-tuning on the target data.

## Results

From `FINAL_run/ipynbRun/output/models/*/*_eval_result.txt` (held-out test set of 7,627 epochs; precision, recall and F1 are weighted averages):

| Model | Precision | Recall | F1 | Cohen's kappa | AUC |
|---|---|---|---|---|---|
| Residual CNN + BiGRU + attention | 0.842 | 0.845 | 0.843 | 0.788 | 0.976 |
| Two-branch multi-scale CNN | 0.834 | 0.841 | 0.835 | 0.780 | 0.974 |
| CNN + Transformer + MLP | 0.821 | 0.831 | 0.824 | 0.767 | 0.966 |
| U-Sleep | 0.816 | 0.830 | 0.818 | 0.765 | 0.969 |
| SleepStagerChambon2018 | 0.784 | 0.805 | 0.781 | 0.728 | 0.957 |
| SleepStagerBlanco2020 | 0.578 | 0.651 | 0.596 | 0.485 | 0.883 |

## Running

```bash
pip install torch braindecode mne numpy scikit-learn tqdm matplotlib
cd FINAL_run
python main.py
```

The dataset is not included. Set `DATA_FOLDER` in `dataloader.py` to a directory containing the `SleepSource/`, `LeaderboardSleep/` and `finalSleep/` folders of `.npy` recordings and labels.
