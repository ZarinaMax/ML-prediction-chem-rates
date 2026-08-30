import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(1)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_LEVELS_BY_REACTION = {
    'O2O': 36, 'O2O2': 36, 'N2N': 47, 'N2O': 47, 'N2N2': 47, 'NON': 39, 'NOO': 39,
}

df = pd.read_csv('dataset.csv')
target_cols = [c for c in df.columns if c.startswith('log10_k_diss')]
reaction_cols = [c for c in df.columns if c.startswith('reaction_')]

df['invT'] = 1.0 / df['temperature_K']
feature_cols = ['temperature_K', 'invT', 'zero_vibr_energy'] + reaction_cols

X = df[feature_cols].values
y = df[target_cols].values

mask = np.zeros_like(y, dtype=bool)
for rcol in reaction_cols:
    rname = rcol.replace('reaction_', '')
    n_valid = N_LEVELS_BY_REACTION[rname]
    rows = df[rcol].values.astype(bool)
    mask[rows, :n_valid] = True

X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
    X, y, mask, test_size=0.33, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
    X_temp, y_temp, mask_temp, test_size=0.5, random_state=RANDOM_STATE)

x_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_s = x_scaler.fit_transform(X_train)
X_val_s = x_scaler.transform(X_val)
X_test_s = x_scaler.transform(X_test)

y_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_s = y_scaler.fit_transform(y_train)
y_val_s = y_scaler.transform(y_val)
y_test_s = y_scaler.transform(y_test)

X_train_t = torch.tensor(X_train_s, dtype=torch.float64)
X_val_t = torch.tensor(X_val_s, dtype=torch.float64)
X_test_t = torch.tensor(X_test_s, dtype=torch.float64)

y_train_t = torch.tensor(y_train_s, dtype=torch.float64)
y_val_t = torch.tensor(y_val_s, dtype=torch.float64)

mask_train_t = torch.tensor(mask_train, dtype=torch.float64)
mask_val_t = torch.tensor(mask_val, dtype=torch.float64)

scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float64)
min_t = torch.tensor(y_scaler.min_, dtype=torch.float64)

INPUT_DIM = X_train_t.shape[1]
N_OUTPUTS = y_train_t.shape[1]
HIDDEN_SIZE = 100

class MultiHeadFNN(nn.Module):
    def __init__(self, input_dim, hidden_size, n_outputs):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh()
        )
        self.head = nn.Linear(hidden_size, n_outputs)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(x))

def masked_mse(pred, target, m):
    diff2 = (pred - target) ** 2 * m
    return diff2.sum() / m.sum()

def masked_mspe_pt(pred_scaled, target_scaled, m):
    pred_u = (pred_scaled - min_t) / scale_t
    target_u = (target_scaled - min_t) / scale_t
    err2 = ((target_u - pred_u) / target_u) ** 2
    return (err2 * m).sum() / m.sum()

def masked_mape_log(y_true, y_pred, m):
    err = np.abs(y_true - y_pred) / np.abs(y_true)
    return (err * m).sum() / m.sum() * 100

model = MultiHeadFNN(INPUT_DIM, HIDDEN_SIZE, N_OUTPUTS).double()

optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Phase 1: training with Adam (5000 epochs)...")
model.train()
for epoch in range(5000):
    optimizer_adam.zero_grad()
    pred = model(X_train_t)
    loss = masked_mse(pred, y_train_t, mask_train_t)
    loss.backward()
    optimizer_adam.step()

optimizer_lbfgs_mse = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=5000, 
    tolerance_grad=1e-11, tolerance_change=1e-13, 
    history_size=300, line_search_fn="strong_wolfe"
)
print("Phase 2: full-batch 64-bit L-BFGS (MSE)...")
def closure_mse():
    optimizer_lbfgs_mse.zero_grad()
    loss = masked_mse(model(X_train_t), y_train_t, mask_train_t)
    loss.backward()
    return loss
for step in range(2):
    optimizer_lbfgs_mse.step(closure_mse)

optimizer_lbfgs_mape = torch.optim.LBFGS(
    model.parameters(), lr=0.5, max_iter=5000, 
    tolerance_grad=1e-11, tolerance_change=1e-13, 
    history_size=300, line_search_fn="strong_wolfe"
)
print("Phase 3: Fine-tuning exact metric with L-BFGS (MSPE)...")
def closure_mape():
    optimizer_lbfgs_mape.zero_grad()
    loss = masked_mspe_pt(model(X_train_t), y_train_t, mask_train_t)
    loss.backward()
    return loss
for step in range(2):
    optimizer_lbfgs_mape.step(closure_mape)

# Evaluation
model.eval()
with torch.no_grad():
    val_pred = model(X_val_t)
    pred_unscaled = y_scaler.inverse_transform(val_pred.numpy())
    mape = masked_mape_log(y_val, pred_unscaled, mask_val)
    print(f"\nValidation log10(k) MAPE = {mape:.4f}%")

    pred_test_s = model(X_test_t).numpy()
    pred_test = y_scaler.inverse_transform(pred_test_s)
    final_mape = masked_mape_log(y_test, pred_test, mask_test)
    print(f"Final log10(k) MAPE on test set: {final_mape:.4f}%")

traced_model = torch.jit.trace(model, X_test_t[:1])
start = time.perf_counter()
runs = 1000
with torch.no_grad():
    for _ in range(runs):
        traced_model(X_test_t[:1])
print(f"Inference: {(time.perf_counter()-start)/runs*1000:.4f} ms/call")
