
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#https://practicaldatascience.co.uk/data-science/how-to-visualise-data-using-boxplots-in-seaborn
#data_model = 'experiments_timespan_moodle.csv'
data_model = 'experiments_timespan_1trimester.csv'

# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)
fig, ax1 = plt.subplots(figsize=(7,5))
#precision
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(y='precision', x='balancing',
                 data=df_model,
                 ax=ax1,
                 palette=["m", "g"],#palette="husl", #"bright", palette="colorblind", palette="Set2","husl", ["m", "g"], "muted", "deep","pastel"
                 hue="dataset",
                 width=0.7
                )

plt.show()

'''
#recall
sns.boxplot(y='recall', x='balancing',
                 data=df_model,
                 palette=["m", "g"], #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
plt.show()

#accuracy
sns.boxplot(y='accuracy', x='balancing',
                 data=df_model,
                 palette=["m", "g"], #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
plt.show()

#inspection_rate
sns.boxplot(y='inspection_rate', x='balancing',
                 data=df_model,
                 palette=["m", "g"], #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
plt.show()

#F1-score
sns.boxplot(y='f1_score', x='balancing',
                 data=df_model,
                 palette=["m", "g"], #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
plt.show()

#mcc
sns.boxplot(y='mcc', x='balancing',
                 data=df_model,
                 palette=["m", "g"], #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
plt.show()
'''