import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["OMP_NUM_THREADS"] = "1"

import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
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

X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
    X, y, mask, test_size=0.15, random_state=RANDOM_STATE)

x_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_s = x_scaler.fit_transform(X_train)
X_test_s = x_scaler.transform(X_test)

y_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_s = y_scaler.fit_transform(y_train)
y_test_s = y_scaler.transform(y_test)

y_train_s = y_train_s * mask_train
y_test_s = y_test_s * mask_test

INPUT_DIM = X_train_s.shape[1]
N_OUTPUTS = y_train_s.shape[1]

model = keras.models.Sequential([
    keras.layers.Dense(100, activation="tanh", input_shape=(INPUT_DIM,)),
    keras.layers.Dense(N_OUTPUTS)
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mse")

def masked_mape_log(y_true, y_pred, m):
    err = np.abs(y_true - y_pred) / np.abs(y_true)
    return (err * m).sum() / m.sum() * 100

print("Training Keras FNN (3000 epochs)...")
history = model.fit(
    X_train_s, y_train_s,
    batch_size=128,
    epochs=3000,
    verbose=0,
    validation_data=(X_test_s, y_test_s)
)

pred_test_s = model.predict(X_test_s, verbose=0)
pred_test = y_scaler.inverse_transform(pred_test_s)
final_mape = masked_mape_log(y_test, pred_test, mask_test)

print(f"\nFinal log10(k) MAPE on test set: {final_mape:.4f}%")

@tf.function(reduce_retracing=True)
def fast_predict(x):
    return model(x, training=False)

_ = fast_predict(X_test_s[0:1]) 

start = time.perf_counter()
runs = 1000
for _ in range(runs):
    _ = fast_predict(X_test_s[0:1])
per_call_ms = (time.perf_counter() - start) / runs * 1000

print(f"Inference: {per_call_ms:.4f} ms/call")
