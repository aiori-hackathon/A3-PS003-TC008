import numpy as np

mean, std = np.load("data/scale.npy")
for j in [1, 2, 3, 4]:
    d = np.load(f"data/junction_{j}.npz")
    pred = d["X_test"][:, -1]                  # last hour as the guess
    err = (pred - d["y_test"]) * std
    print(f"Junction {j}: naive MAE={np.abs(err).mean():.2f}  RMSE={np.sqrt((err**2).mean()):.2f}")