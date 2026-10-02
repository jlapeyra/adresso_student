import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os

df_list = []

# # Load the different result runs
# df1 = pd.read_csv("adresso_student/outputs/results_jul30/all_results.csv")
# df1["link_features"] = "plain"
# df1["ml"] = "original"
# df_list.append(df1)

df2 = pd.read_csv("adresso_student/outputs/results_sep10/results_so_far.csv")
df2["ml"] = "original"
df_list.append(df2)

for fn in glob("adresso_student/outputs/extended_ml_results/results_*"):
    if "all_results.csv" in os.listdir(fn):
        fn = os.path.join(fn, "all_results.csv")
    elif "results_so_far.csv" in os.listdir(fn):
        fn = os.path.join(fn, "results_so_far.csv")
    df_temp = pd.read_csv(fn)
    if "link_features" not in df_temp.columns:
        df_temp["link_features"] = "both"
    df_temp["ml"] = "extended"
    df_list.append(df_temp)


# Join all runs
df = pd.concat(df_list, ignore_index=True)

df["link_features"] = df["link_features"].replace("clinical", "plain")


df["ablation"] = pd.Categorical(
    df["ablation"],
    categories=df["ablation"].drop_duplicates().tolist(),
    ordered=True
)

df["link_features"] = pd.Categorical(
    df["link_features"],
    categories=df["link_features"].drop_duplicates().tolist(),
    ordered=True
)


# Plot
fig, ax = plt.subplots()

sns.pointplot(
    data=df[df["ml"] == "original"],
    x="ablation",
    y="balanced_accuracy",
    hue="link_features",
    dodge=0.2,
    alpha=1.0,
    capsize=0.1,
    ax=ax,
)

sns.pointplot(
    data=df[df["ml"] == "extended"],
    x="ablation",
    y="balanced_accuracy",
    hue="link_features",
    dodge=0.4,
    alpha=0.4,
    capsize=0.1,
    ax=ax,
    legend=False
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()