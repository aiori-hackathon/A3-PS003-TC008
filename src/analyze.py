import numpy as np, pandas as pd
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2]
LAST = 5                       # average the last 5 rounds/epochs to smooth the wobble

def load(pattern):
    out = []
    for s in SEEDS:
        try:
            out.append(pd.read_csv(pattern.format(s=s)))
        except FileNotFoundError:
            print("missing:", pattern.format(s=s))
    return out

runs = {
    "Centralised":    load("results_central_seed{s}.csv"),
    "Federated IID":  load("results_fed_iid_seed{s}.csv"),
    "Federated non-IID": load("results_fed_noniid_seed{s}.csv"),
}

# ---------- summary table ----------
lines = []
print(f"\n{'Setup':20s} {'MAE (mean±sd)':18s} {'RMSE (mean±sd)':18s} per-junction MAE (mean)")
for name, dfs in runs.items():
    if not dfs: continue
    mae  = np.array([d["mae_all"].tail(LAST).mean() for d in dfs])
    rmse = np.array([d["rmse_all"].tail(LAST).mean() for d in dfs])
    pj   = np.array([[d[f"mae_j{j}"].tail(LAST).mean() for j in range(1, 5)] for d in dfs]).mean(0)
    print(f"{name:20s} {mae.mean():.3f} ± {mae.std():.3f}     {rmse.mean():.3f} ± {rmse.std():.3f}     {np.round(pj, 2)}")
    lines.append([name, mae.mean(), mae.std(), rmse.mean(), rmse.std(), *pj])
pd.DataFrame(lines, columns=["setup", "mae_mean", "mae_sd", "rmse_mean", "rmse_sd",
                             "mae_j1", "mae_j2", "mae_j3", "mae_j4"]).to_csv("summary.csv", index=False)

# ---------- convergence: how many rounds to get within 5% of the final MAE ----------
print("\nConvergence (first round/epoch where MAE <= 1.05 x final MAE, mean over seeds):")
for name, dfs in runs.items():
    if not dfs: continue
    hits = []
    for d in dfs:
        final = d["mae_all"].tail(LAST).mean()
        hits.append(int((d["mae_all"] <= 1.05 * final).idxmax()) + 1)
    print(f"  {name:20s} {np.mean(hits):.1f}")

# ---------- communication ----------
for name in ["Federated IID", "Federated non-IID"]:
    if runs[name]:
        d = runs[name][0]
        print(f"\n{name}: {int(d['bytes_round'].iloc[0])} bytes/round, "
              f"{int(d['bytes_cum'].iloc[-1])} bytes after {len(d)} rounds")

colors = {"Centralised": "tab:green", "Federated IID": "tab:blue", "Federated non-IID": "tab:red"}

# ---------- graph 1: MAE vs round ----------
plt.figure(figsize=(8, 5))
for name, dfs in runs.items():
    if not dfs: continue
    M = np.array([d["mae_all"].values for d in dfs])
    x = np.arange(1, M.shape[1] + 1)
    plt.plot(x, M.mean(0), label=name, color=colors[name])
    plt.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), alpha=0.2, color=colors[name])
plt.xlabel("Round (federated) / Epoch (centralised)"); plt.ylabel("Test MAE (vehicles/hour)")
plt.title("Convergence (mean ± sd over 3 seeds)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("graph1_mae_vs_round.png", dpi=150); plt.close()

# ---------- graph 2: MAE vs cumulative bytes ----------
plt.figure(figsize=(8, 5))
for name in ["Federated IID", "Federated non-IID"]:
    if not runs[name]: continue
    M = np.array([d["mae_all"].values for d in runs[name]]).mean(0)
    kb = runs[name][0]["bytes_cum"].values / 1024
    plt.plot(kb, M, label=name, color=colors[name])
plt.xlabel("Cumulative communication (KB)"); plt.ylabel("Test MAE (vehicles/hour)")
plt.title("Accuracy vs communication cost"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("graph2_mae_vs_bytes.png", dpi=150); plt.close()

# ---------- graph 3: per-junction MAE ----------
plt.figure(figsize=(8, 5))
w = 0.25
for i, name in enumerate(runs):
    if not runs[name]: continue
    pj = np.array([[d[f"mae_j{j}"].tail(LAST).mean() for j in range(1, 5)] for d in runs[name]])
    plt.bar(np.arange(4) + (i - 1) * w, pj.mean(0), w, yerr=pj.std(0), label=name, color=colors[name], capsize=3)
plt.xticks(range(4), ["Junction 1", "Junction 2", "Junction 3", "Junction 4"])
plt.ylabel("Test MAE (vehicles/hour)"); plt.title("Per-junction error"); plt.legend()
plt.tight_layout(); plt.savefig("graph3_per_junction.png", dpi=150); plt.close()

print("\nSaved: summary.csv, graph1_mae_vs_round.png, graph2_mae_vs_bytes.png, graph3_per_junction.png")