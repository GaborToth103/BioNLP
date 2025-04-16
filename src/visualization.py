import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data/output/model_metrics.csv")

# Set the figure size
plt.figure(figsize=(12, 6))

# Bar width and position indices
bar_width = 0.25
index = range(len(df))

# Plot Precision, Recall, and F1 Score as grouped bars
plt.bar([i - bar_width for i in index], df['mean_precision'], width=bar_width, label='Precision')
plt.bar(index, df['mean_recall'], width=bar_width, label='Recall')
plt.bar([i + bar_width for i in index], df['mean_f1'], width=bar_width, label='F1 Score')

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Evaluation Metrics')
plt.xticks(index, df['model_name'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("data/output/model_metrics_plot.png", dpi=300, bbox_inches='tight')
