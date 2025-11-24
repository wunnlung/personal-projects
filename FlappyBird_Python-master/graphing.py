import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv("training_data.csv")

# Basic plot
plt.figure(figsize=(10, 5))
plt.plot(df["generation"], df["score"], label="Best Score per Generation", color='dodgerblue')

# Optional smoothing (rolling average)
if len(df) >= 10:
    df["smoothed"] = df["score"].rolling(window=10).mean()
    plt.plot(df["generation"], df["smoothed"], label="10-Gen Rolling Avg", color='orange', linestyle='--')

plt.title("Flappy Bird AI Training Progress")
plt.xlabel("Generation")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
