
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", palette="pastel")
data_url = 'experiments_kmeans_cmeans.csv'
# read data from url as pandas dataframe
gapminder = pd.read_csv(data_url)
# subset the gapminder data frame for rows with year values 1952 and 2007
df1 = gapminder[gapminder['model'].isin(["kmeans","cmeans"])]
sns.boxplot(y='mcc', x='balancing',
                 data=df1,
                 palette="colorblind", #palette=["m", "g"]
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()