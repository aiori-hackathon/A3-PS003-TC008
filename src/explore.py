import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traffic.csv")

print("COLUMNS:", list(df.columns))
print(df.head(), "\n")
df.info()

df["DateTime"] = pd.to_datetime(df["DateTime"])
print("\nOverall dates:", df["DateTime"].min(), "to", df["DateTime"].max())

print("\nPer junction:")
print(df.groupby("Junction")["DateTime"].agg(["min", "max", "count"]))
print("\nVehicles stats per junction:")
print(df.groupby("Junction")["Vehicles"].describe())
print("\nMissing values:\n", df.isna().sum())

fig, ax = plt.subplots(figsize=(12, 5))
for j, g in df.groupby("Junction"):
    g = g.sort_values("DateTime").head(336)
    ax.plot(g["DateTime"], g["Vehicles"], label=f"Junction {j}")
ax.set_ylabel("Vehicles per hour")
ax.legend()
plt.tight_layout()
plt.savefig("junctions.png")
print("\nSaved junctions.png")