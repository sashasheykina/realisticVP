
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

my_colors = ["#3498db",         #"#9b59b6",
             "#2ecc71", "#006a4e"]
sns.set_theme(style="ticks", palette="pastel")
data_model = 'experiments_timespan.csv'
# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)

#precision
my_pr = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='precision',
                 data=my_pr, palette="Set2",#palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()
'''
#mcc
my_mcc = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='mcc',
                 data=my_mcc, #palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()

#recall
my_recall = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='recall',
                 data=my_recall, #palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()

#accuracy
my_acc = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='accuracy',
                 data=my_acc, #palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()

#F1-score
my_f1 = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='f1_score',
                 data=my_f1, #palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()

#inspection_rate
my_ir = df_model[df_model['dataset'].isin(["moodle"])]
sns.boxplot(x='balancing', y='inspection_rate',
                 data=my_ir, #palette="colorblind",palette=["m", "g"],palette="Set2"
                 hue='dataset')
sns.despine(offset=10, trim=True)
plt.show()
'''