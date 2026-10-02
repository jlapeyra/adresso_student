import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os


df = pd.read_csv("result_analysis/results_text.csv")


sns.pointplot(
    data=df,
    x="text_encoder",
    y="balanced_accuracy",
    hue="link_features",
    dodge=0.2,
    capsize=0.1,
)


plt.xticks(rotation=10)
plt.tight_layout()
plt.show()