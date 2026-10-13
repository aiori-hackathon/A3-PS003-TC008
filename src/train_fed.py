import sys, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import GRUNet

mode = sys.argv[1]               # "iid" or "noniid"
seed = int(sys.argv[2])
ROUNDS, LOCAL_EPOCHS, BATCH, LR = 30, 1, 64, 1e-3

torch.manual_seed(seed); np.random.seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = np.load("data/scale.npy")
D = {j: np.load(f"data/junction_{j}.npz") for j in [1, 2, 3, 4]}

# ---- build the 4 clients ----
if mode == "noniid":                       # one junction = one client
    clients = [(D[j]["X_train"], D[j]["y_train"]) for j in D]
else:                                      # pool training data, shuffle, deal out evenly
    X = np.concatenate([D[j]["X_train"] for j in D])
    y = np.concatenate([D[j]["y_train"] for j in D])
    parts = np.array_split(np.random.permutation(len(X)), 4)
    clients = [(X[p], y[p]) for p in parts]
sizes = [len(c[0]) for c in clients]
print("client sizes:", sizes)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X).to(dev)).cpu().numpy()
    err = (pred - y) * std
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())

def local_train(global_state, X, y):
    model = GRUNet().to(dev)
    model.load_state_dict(global_state)            # client receives global model
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                        batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            nn.MSELoss()(model(xb), yb).backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def fedavg(states, sizes):                         # weighted average of weights
    total = sum(sizes)
    return {k: sum(s[k] * (n / total) for s, n in zip(states, sizes)) for k in states[0]}

glob = GRUNet()
global_state = {k: v.clone() for k, v in glob.state_dict().items()}
model_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
print("bytes per model copy:", model_bytes)

Xte_all = np.concatenate([D[j]["X_test"] for j in D])
yte_all = np.concatenate([D[j]["y_test"] for j in D])
eval_model = GRUNet().to(dev)

rows, cum = [], 0
for rnd in range(1, ROUNDS + 1):
    states = [local_train(global_state, X, y) for X, y in clients]
    global_state = fedavg(states, sizes)

    round_bytes = model_bytes * len(clients) * 2   # 4 downloads + 4 uploads
    cum += round_bytes

    eval_model.load_state_dict(global_state)
    per_j = [evaluate(eval_model, D[j]["X_test"], D[j]["y_test"])[0] for j in D]
    mae, rmse = evaluate(eval_model, Xte_all, yte_all)
    rows.append([rnd, *per_j, mae, rmse, round_bytes, cum])
    print(f"round {rnd:2d}  MAE_all={mae:.3f}  RMSE_all={rmse:.3f}  "
          f"per-junction={np.round(per_j, 2)}  bytes={round_bytes}")

fn = f"results_fed_{mode}_seed{seed}.csv"
with open(fn, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "mae_j1", "mae_j2", "mae_j3", "mae_j4",
                "mae_all", "rmse_all", "bytes_round", "bytes_cum"])
    w.writerows(rows)
print("Saved", fn)