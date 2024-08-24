import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame
losses_df = pd.read_csv("results/epoch_losses.csv")
print(losses_df)

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
plt.title("Metrics vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend(title="Metric")
plt.grid(True)

plt.savefig("results/all_losses.png", dpi=300)

# Show plot
plt.show()

last_run = losses_df.iloc[-1]
print(f"Last metrics: \n{last_run}")
