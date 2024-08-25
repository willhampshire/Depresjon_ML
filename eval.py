import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame
losses_df = pd.read_csv("results/epoch_losses.csv")
print(losses_df)

# losses_df = losses_df.drop(["deltamadrs_mae", "loss", "madrs2_mae"], axis=1)

# Melt the DataFrame to long format
df_melted = losses_df.melt(id_vars="epoch", var_name="Metric", value_name="Value")

sns.set_palette("muted")
sns.set_context("notebook")

# Plot using seaborn with scatter style
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_melted,
    x="epoch",
    y="Value",
    hue="Metric",
    marker="o",
    linestyle="-",
    markersize=8,
)

# Customize the plot
plt.title("Loss (MSE) & validation loss (MAE) vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss [MADRS]")
plt.legend(title="Metric")
plt.grid(True)

plt.ylim(0, 0.05)
plt.xticks(losses_df["epoch"].astype(int))

plt.savefig("results/all_losses.png", dpi=300)

# Show plot
plt.show()


scores = pd.read_csv("data/scores.csv")
range_madrs2 = scores["madrs2"].max() - scores["madrs2"].min()


last_run = losses_df.iloc[-1]
print(f"Last metrics: \n{last_run}")

# print(scores["madrs2"].max(), scores["madrs2"].min()) check max-min

print(
    f"\nValidation losses (input madrs range {range_madrs2:.1f})"
    f"\ndelta MAE: {last_run['val_deltamadrs_mae']:.5f}, {last_run['val_deltamadrs_mae']/range_madrs2 * 100:.4f}%"
    f"\nmadrs2 MAE: {last_run['val_madrs2_mae']:.5f}, {last_run['val_deltamadrs_mae']/range_madrs2 * 100:.4f}%"
)
