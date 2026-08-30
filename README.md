# Prediction of State-to-State Dissociation Rate Coefficients Using Machine-Learning Algorithms

This repository contains the dataset and machine-learning pipeline accompanying the paper:
**"Prediction of State-to-State Dissociation Rate Coefficients Using Machine-Learning Algorithms"**
*Vestnik St. Petersburg University, Mathematics, 2024.*

*Note: This is a lightweight, reproducible demonstration of the paper's methodology.
The paper used ~100²=10,000 parameter vectors for the vibrational-excitation-only 
case (n=3) before the train/test split; this repo's `dataset.csv` uses a coarser 
step (`T_step=50`, ~1,300 rows) for fast local execution. Expect MAPE and inference 
values to be close but not identical to the paper's Table 1 — the relative model 
ranking is preserved: Decision Tree remains the weakest performer, while FNN and 
k-NN both achieve comparably low, sub-percent MAPE, matching the paper's conclusion 
that FNN is the most practical choice overall.*

**DOIs:**
- English version: [10.1134/S1063454124700390](https://doi.org/10.1134/S1063454124700390)
- Russian version: [10.21638/spbu01.2024.413](https://doi.org/10.21638/spbu01.2024.413)

## Overview

This work applies machine learning (k-NN, Decision Trees, and neural networks)
to approximate state-to-state dissociation rate coefficients — a quantity that
is expensive to compute from first-principles kinetic theory and is used in
modeling nonequilibrium gas flows. 
The paper was presented at open-science international events like ICMAR 2024 
(International Conference on the Methods of Aerophysical Research) 
and CMMASS 2023 (International Conference on Computational Mechanics and Modern Applied Software Systems).

## Scope of This Repository

- **No physical solver is implemented here.** This repository does not
  simulate any physical system. The numerical values in `dataset.csv` were
  generated separately as part of the theoretical work described in the
  paper, using standard kinetic theory formulas.
- **This repository is a regression/optimization pipeline.** It takes a table
  of precomputed numerical input–output pairs and fits generic supervised
  learning models (k-NN, Decision Tree, FNN) to approximate the mapping
  between them. The code itself has no dependency on what the columns
  physically represent — it would run identically on any tabular regression
  dataset of the same shape.
- The contribution demonstrated here is that generic ML methods can
  approximate a known, precomputed numerical mapping much faster than
  recomputing it analytically each time — a computational optimization
  technique, applicable to any sufficiently expensive-to-evaluate function.

## Repository Structure
- `dataset.csv`: Pre-computed dataset containing input features (temperature,
  vibrational levels, reaction type) and target reaction rate coefficients.
- `Classical_ML.py`: Data preprocessing (scaling, log-transformation of
  targets) and baseline classical ML models (k-NN, Decision Tree).
- `Deep_Learning_pytorch.py` & `Deep_Learning_keras.py`: Neural network models for sequence-based
  prediction of rate coefficients, demonstrating multiple backend approaches.

## Technologies Used
- Python 3.9
- scikit-learn
- Pytorch & Tensorflow / Keras
- Pandas & NumPy

## Notes on Methodology

Target values span many orders of magnitude, which required
log-transformation of the targets to make the regression problem tractable
and avoid vanishing-gradient issues during training. This preprocessing
approach, along with the Feedforward Neural Network (FNN) used here,
generalizes to other domains involving wide-dynamic-range time-series data.

## How to Cite This Work

**English Edition:**
```bibtex
@article{maksudova2024prediction_eng,
  title={Prediction of State-to-State Dissociation Rate Coefficients Using Machine-Learning Algorithms},
  author={Maksudova, Z. M. and Savelev, A. S. and Kustova, E. V.},
  journal={Vestnik St. Petersburg University, Mathematics},
  volume={57},
  number={4},
  pages={584--592},
  year={2024},
  publisher={Pleiades Publishing, Ltd.},
  doi={10.1134/S1063454124700390}
}
```

**Russian Edition:**
```bibtex
@article{maksudova2024prediction_ru,
  title={Прогнозирование поуровневых коэффициентов скорости диссоциации при помощи алгоритмов машинного обучения},
  author={Максудова, З. М. and Савельев, А. С. and Кустова, Е. В.},
  journal={Вестник Санкт-Петербургского университета. Математика. Механика. Астрономия},
  volume={11},
  number={4},
  pages={782--793},
  year={2024},
  publisher={Санкт-Петербургский государственный университет},
  doi={10.21638/spbu01.2024.413}
}
```

## License
This project is licensed under the MIT License — see the LICENSE file for details.
