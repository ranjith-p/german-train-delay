# 🚆 German Train Delay Prediction Benchmark

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

This repository builds an open, reproducible benchmark for short-horizon train delay prediction in Germany, comparing four model architectures under one shared protocol.

---

## 📌 What this does

Given a snapshot of a train's journey so far, using its recent delay history, current station, and time, the task is to predict its delay at each of the **next ten stops**. Here's how a snapshot is built and used:

```
DB raw data + weather + infrastructure data
                    |
                    v
            Snapshot creation
                    |
        +-----------+------------+
        |                        |
        v                        v
   Past stops              10 Future stops
  (input features)           (targets)
        |                        |
        v                        v
  Model training              Prediction
```

At a randomly sampled point in an ongoing trip, everything that already happened becomes the input (features), and everything still ahead becomes the ten targets the model has to predict. The same snapshot logic is shared by all four architectures below:

| Model | Type |
|---|---|
| 🌳 **XGBoost** | Tree-based |
| 🧠 **MLP** | Feedforward neural network |
| 🔁 **LSTM** | Sequential neural network |
| 🕸️ **GNN** | Graph neural network (railway station topology) |

All four are benchmarked against a translation baseline (simply carrying the last observed delay forward).

## 🗂️ Data sources

All data used is openly licensed (CC BY 4.0):

| Source | What it provides |
|---|---|
| [Deutsche Bahn open data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (via Brömmel, Hugging Face) | Train operational data — scheduled and actual arrival/departure times |
| [Open-Meteo Historical Weather API](https://open-meteo.com/) | Hourly weather observations per station |
| [DB InfraGO Streckennetz](https://mobilithek.info/offers/922109165921083392) | Track infrastructure — electrification, track count, speed limits |

**Coverage:** Berlin, Hamburg, and München · **Train:** Nov 2025–Apr 2026 · **Test:** May–Jul 2026 (forward-in-time, no overlap)

## ⚙️ Pipeline

```
Raw data  →       Feature engineering  →   Model training & evaluation
   │                  │                         │
1_raw_data_prep   2_feature_engineering    3_model_training
```

| Notebook | What it does |
|---|---|
| `1_raw_data_prep.ipynb` | Downloads and cleans the raw train, weather, and infrastructure data; resolves trip segmentation (train IDs repeat daily, so this reconstructs actual individual journeys); builds the snapshot-based training and test sets |
| `2_feature_engineering.ipynb` | Builds the candidate feature groups (infrastructure, electrification, timetable, congestion, cyclical time, weather, past delay history) and the station graph used by the GNN |
| `3_model_training.ipynb` | Trains and tunes all four models, runs feature selection (greedy backward elimination) and the topology ablation, and produces the evaluation results |

## 🏆 Key results

Pooled test-set performance across all ten prediction horizons:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Translation baseline | 2.354 | 9.289 | 0.339 |
| XGBoost | 1.809 | 8.126 | 0.494 |
| MLP | 1.617 | 8.391 | 0.461 |
| LSTM | 1.766 | 8.725 | 0.417 |
| **GNN** 🥇 | **1.571** | **7.330** | **0.589** |

The GNN achieves the best pooled result on every metric. Full per-horizon results, the weather and feature-selection findings, and the topology ablation are detailed in the dissertation.

## 🔧 Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- XGBoost
- PyTorch (MLP, LSTM, GNN)
- Optuna (hyperparameter tuning)

## 📝 Notes

- Notebooks are numbered and intended to be run in order; each stage writes its output for the next to consume.
