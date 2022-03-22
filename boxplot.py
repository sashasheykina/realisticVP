
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

my_colors = ["#3498db",         #"#9b59b6",
             "#2ecc71", "#006a4e"]
sns.set_theme(style="ticks", palette="pastel")
data_model = 'experiments_moodle.csv'
# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)
#precision
'''my_pr = df_model[df_model['model'].isin(["kmeans","random_forest"])]
sns.boxplot(y='precision', x='test_set_release',
                 data=my_pr,
                 palette="Set2",
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()'''
#mcc %vulnerability
'''my_mcc = df_model[df_model['model'].isin(["random_forest"])]
sns.boxplot(y='mcc', x='%vulnerability',
                 data=my_mcc,
                 palette="Set2",#palette=["m", "g"],
                 hue='model')
sns.despine(offset=10, trim=True)
plt.xticks(rotation = 45, ha = 'right')
plt.show()'''

''' 
my_pr = df_model[df_model['model'].isin(["random_forest"])]
sns.set_palette( my_colors )
sns.boxplot(y='precision', x='%vulnerability',
                 data=my_pr,
                 hue='model')
sns.despine(offset=10, trim=True)
plt.xticks(rotation = 45, ha = 'right')
plt.show()'''
'''my_mcc = df_model[df_model['model'].isin(["kmeans","random_forest"])]
sns.boxplot(y='mcc', x='balancing',
                 data=my_mcc,
                 palette="colorblind",
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()'''
#recall
'''my_recall = df_model[df_model['model'].isin(["kmeans","random_forest"])]
sns.boxplot(y='recall', x='test_set_release',
                 data=my_recall,
                 palette="Set2",#palette=["m", "g"],
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()'''
#f1_score
my_f1 = df_model[df_model['model'].isin(["kmeans","random_forest"])]
sns.boxplot(y='f1_score', x='balancing',
                 data=my_f1,
                 palette="colorblind",
                 hue='model')
sns.despine(offset=10, trim=True)
plt.show()