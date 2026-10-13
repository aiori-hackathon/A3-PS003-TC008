import os
import numpy as np
import pandas as pd

WINDOW = 12                      # model looks at the past 12 hours
CUT = pd.Timestamp("2017-05-01") # before = train, from here on = test

df = pd.read_csv("traffic.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values(["Junction", "DateTime"])

# scaling numbers come from TRAINING data only (no peeking at the future)
train_all = df[df["DateTime"] < CUT]["Vehicles"]
mean, std = train_all.mean(), train_all.std()
print(f"Scaling: mean={mean:.2f}, std={std:.2f}")

def make_windows(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])   # past 12 hours
        y.append(values[i + window])     # the next hour (what we predict)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

os.makedirs("data", exist_ok=True)

for j in [1, 2, 3, 4]:
    g = df[df["Junction"] == j]
    train_vals = g[g["DateTime"] < CUT]["Vehicles"].values
    # test starts WINDOW hours earlier so the first test prediction has history
    test_vals = g[g["DateTime"] >= CUT - pd.Timedelta(hours=WINDOW)]["Vehicles"].values

    train_vals = (train_vals - mean) / std
    test_vals = (test_vals - mean) / std

    X_tr, y_tr = make_windows(train_vals, WINDOW)
    X_te, y_te = make_windows(test_vals, WINDOW)

    np.savez(f"data/junction_{j}.npz",
             X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
    print(f"Junction {j}: train {X_tr.shape}, test {X_te.shape}")

np.save("data/scale.npy", np.array([mean, std]))
print("Done. Files saved in the data folder.")