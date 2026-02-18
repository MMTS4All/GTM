# GTM: A General Time-series Model

[![Paper](https://img.shields.io/badge/Paper-arXiv:2412.03068-B31B1B.svg)](https://arxiv.org/abs/2412.03068)
[![GitHub](https://img.shields.io/github/stars/MMTS4All/GTM?style=social)](https://github.com/MMTS4All/GTM)

> **GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data**  
> _Published as a conference paper at ICLR 2026_

GTM (General Time-series Model) is a foundation model for time series analysis that advances representation learning via a novel frequency-domain attention mechanism and a unified pre-training strategy. GTM is the first generative-task-agnostic model for time series, enabling seamless adaptation to various generative tasks without any task-specific modifications.

## 🌟 Key Features

- **Frequency-Domain Attention**: Novel Fourier attention mechanism that captures time-granularity-aware features
- **Hybrid Pre-training**: Unified reconstruction and autoregressive objectives through hybrid masking
- **Generative-Task-Agnostic**: Seamless adaptation to forecasting, imputation, and anomaly detection without modifications
- **Scalable Architecture**: Follows scaling laws with performance improving as model size and pre-training data increase
- **Multi-Granularity Support**: Explicitly incorporates time granularity for robust representation learning

## 🏗️ Architecture Overview

GTM follows a decoder-only Transformer architecture with specialized components for time series modeling:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GTM Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Time Series Data ──► Input Embedding ──► N-Stack Decoder-only Backbone     │
│                            ▲                                                │
│                            │                                                │
│                     Temporal & Fourier Attention                            │
│                            │                                                │
│                   ┌────────┴────────┐                                      │
│                   │                 │                                      │
│            Temporal Attention  Fourier Attention                            │
│                   │                 │                                      │
│                   └────────┬────────┘                                      │
│                            │                                                │
│                      Decoder Output                                        │
│                            │                                                │
│                            ▼                                                │
│                      Output Projection                                     │
│                            │                                                │
│                            ▼                                                │
│                       Final Output                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Input Embedding**:
   - Reversible Instance Normalization (RevIN)
   - Channel Independence (CI)
   - Patching and masking
   - Linear and positional embeddings

2. **N-Stack Decoder-only Backbone**:
   - Temporal self-attention module
   - Fourier attention module for frequency-domain information
   - Decoder layers with residual connections

3. **Fourier Attention Module**:
   - Captures frequency-specific patterns
   - Time granularity-aware representations
   - Five low-rank modules for different granularities
   - Global frequency learning module

4. **Output Projection**:
   - Unified linear projection layer
   - Instance denormalization

## 🔬 Methodology

### Frequency-Domain Attention Mechanism

GTM introduces a novel Fourier attention mechanism that:
- Transforms temporal patches to frequency domain using FFT
- Learns granularity-aware representations through specialized modules
- Combines multiple frequency patterns using attention weights
- Transforms back to temporal domain using inverse FFT

### Hybrid Pre-training Strategy

Our pre-training framework unifies reconstruction and autoregressive objectives:
- **Random Masking**: Samples patch spans and randomly permutes them
- **Consecutive Tail Masking**: Applies controlled proportion of consecutive masks at sequence tail
- **2D Positional Encoding**: Ensures model awareness of masked span lengths
- **Span Shuffling**: Enhances robustness and generalization

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/MMTS4All/GTM.git
cd GTM

# Create conda environment (recommended)
conda create -n gtm python=3.8
conda activate gtm

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage

### Pre-training

```bash
python run_pretrain.py \
  --task_name pre_train \
  --model_id GTM \
  --data utsd \
  --root_path /path/to/your/data \
  --seq_len 1440 \
  --patch_len 96 \
  --stride 96 \
  --d_model 768 \
  --d_layers 12 \
  --batch_size 1024 \
  --learning_rate 1e-5 \
  --train_epochs 30
```

### Fine-tuning for Forecasting

```bash
python run_forecasting.py \
  --task_name long_term_forecast \
  --model_id GTM \
  --data ETTm1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_model 768 \
  --d_layers 12 \
  --batch_size 32 \
  --learning_rate 1e-4
```

### Supported Tasks

GTM supports multiple time series analysis tasks without architectural modifications:

1. **Long-term Forecasting**: Predict future values in a time series
2. **Imputation**: Fill in missing values in a time series
3. **Anomaly Detection**: Identify anomalous patterns in time series data
4. **Pre-training**: Learn general representations from large-scale time series data

Each task leverages the same model architecture but with different configurations and loss functions.

#### Long-term Forecasting
Predict future values in a time series using historical data. The model takes a sequence of past values and predicts future values for a specified horizon.

#### Imputation
Fill in missing values in a time series. The model is trained to reconstruct missing values by randomly masking portions of the input sequence during training.

#### Anomaly Detection
Identify anomalous patterns in time series data. The model is trained to reconstruct normal patterns and uses reconstruction error to detect anomalies.

#### Pre-training
Learn general representations from large-scale time series data using a hybrid masking strategy that combines random and consecutive tail masking.

## 📈 Performance

GTM consistently outperforms state-of-the-art models across various benchmarks:

| Task | Dataset | GTM MSE | SOTA MSE | Improvement |
|------|---------|---------|----------|-------------|
| Forecasting | ETTh1 | 0.404 | 0.411 | 1.7% |
| Forecasting | ETTm1 | 0.339 | 0.350 | 3.1% |
| Imputation | ETTh1 | 0.053 | 0.055 | 3.6% |
| Anomaly Detection | MSL | 82.53 | 81.92 | +0.61 |

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{he2026gtm,
  title={GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data},
  author={He, Cheng and Huang, Xu and Jiang, Gangwei and Li, Zhaoyi and Lian, Defu and Xie, Hong and Chen, Enhong and Liang, Xijie and Zheng, Zengrong and Lee, Patrick P. C.},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This work was supported by grants from the National Natural Science Foundation of China (Nos. 92367110 and U23A20319).

---

<p align="center">
  Made with ❤️ by the GTM Team
</p>