import sys, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import GRUNet

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed); np.random.seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = np.load("data/scale.npy")

D = {j: np.load(f"data/junction_{j}.npz") for j in [1, 2, 3, 4]}
Xtr = np.concatenate([D[j]["X_train"] for j in D])
ytr = np.concatenate([D[j]["y_train"] for j in D])
loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=64, shuffle=True)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X).to(dev)).cpu().numpy()
    err = (pred - y) * std                    # back to real vehicle counts
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())

model = GRUNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

Xte_all = np.concatenate([D[j]["X_test"] for j in D])
yte_all = np.concatenate([D[j]["y_test"] for j in D])

rows = []
for epoch in range(1, 31):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    per_j = [evaluate(model, D[j]["X_test"], D[j]["y_test"])[0] for j in D]
    mae, rmse = evaluate(model, Xte_all, yte_all)
    rows.append([epoch, *per_j, mae, rmse])
    print(f"epoch {epoch:2d}  MAE_all={mae:.3f}  RMSE_all={rmse:.3f}  per-junction MAE={np.round(per_j, 2)}")

with open(f"results_central_seed{seed}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "mae_j1", "mae_j2", "mae_j3", "mae_j4", "mae_all", "rmse_all"])
    w.writerows(rows)
print("Saved results_central_seed%d.csv" % seed)