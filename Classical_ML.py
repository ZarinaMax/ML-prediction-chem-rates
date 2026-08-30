import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

N_LEVELS_BY_REACTION = {
    'O2O': 36, 'O2O2': 36, 'N2N': 47, 'N2O': 47, 'N2N2': 47, 'NON': 39, 'NOO': 39,
}

df = pd.read_csv('dataset.csv')
target_cols = [c for c in df.columns if c.startswith('log10_k_diss')]
reaction_cols = [c for c in df.columns if c.startswith('reaction_')]

df['invT'] = 1.0 / df['temperature_K']
df['logT'] = np.log(df['temperature_K'])
feature_cols = ['temperature_K', 'invT', 'logT', 'zero_vibr_energy'] + reaction_cols

X = df[feature_cols].values
y = df[target_cols].values

mask = np.zeros_like(y, dtype=bool)
for rcol in reaction_cols:
    rname = rcol.replace('reaction_', '')
    n_valid = N_LEVELS_BY_REACTION[rname]
    rows = df[rcol].values.astype(bool)
    mask[rows, :n_valid] = True

X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
    X, y, mask, test_size=0.33, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def masked_mape_log(y_true, y_pred, m):
    err = np.abs(y_true - y_pred) / np.abs(y_true)
    return (err * m).sum() / m.sum() * 100

def evaluate(name, model):
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mape = masked_mape_log(y_test, pred, mask_test)

    start = time.perf_counter()
    runs = 500
    for _ in range(runs):
        model.predict(X_test_s[:1])
    per_call_ms = (time.perf_counter() - start) / runs * 1000

    print(f"{name}: log10(k) MAPE = {mape:.4f}% | inference: {per_call_ms:.4f} ms/call")

print("k-NN (k=3):")
evaluate("k-NN", KNeighborsRegressor(n_neighbors=3))

print("\nDecision Tree:")
evaluate("DT (max_depth=5)", DecisionTreeRegressor(max_depth=5, min_samples_leaf=3,
                                      splitter="random", random_state=RANDOM_STATE))
evaluate("DT (unlimited)", DecisionTreeRegressor(min_samples_leaf=3,
                                      splitter="random", random_state=RANDOM_STATE))
