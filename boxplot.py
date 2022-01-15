
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", palette="pastel")
data_model = 'experiments_kmeans_cmeans.csv'
# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)
# subset the gapminder data frame for rows with year values 1952 and 2007
my_mcc = df_model[df_model['model'].isin(["kmeans","cmeans"])]
sns.boxplot(y='mcc', x='balancing',
                 data=my_mcc,
                 palette=["m", "g"], #palette="colorblind"
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()
my_recall = df_model[df_model['model'].isin(["kmeans","cmeans"])]
sns.boxplot(y='recall', x='balancing',
                 data=my_recall,
                 palette=["m", "g"],
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()
my_acc = df_model[df_model['model'].isin(["kmeans","cmeans"])]
sns.boxplot(y='accuracy', x='balancing',
                 data=my_acc,
                 palette=["m", "g"],
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()