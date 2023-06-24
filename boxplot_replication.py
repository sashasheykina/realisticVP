import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://practicaldatascience.co.uk/data-science/how-to-visualise-data-using-boxplots-in-seaborn
# data_model = 'experiments_timespan_moodle.csv'
data_model = 'experiments_replication.csv'

# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)
fig, ax1 = plt.subplots(figsize=(7,5))
# precision
sns.set_theme(style="ticks", palette="bright")
sns.boxplot(y='f1_score', x='balancing',
            data=df_model,
            hue="dataset",
            ax=ax1,
            palette=["m", "g"],
            medianprops={'color': 'red', 'label': '_median_'}
            )
ax1.set(ylim=(-0.1, 1))
median_colors = ['m', 'g']
median_lines = [line for line in ax1.get_lines() if line.get_label() == '_median_']
for i, line in enumerate(median_lines):
    line.set_color(median_colors[i % len(median_colors)])

plt.show()