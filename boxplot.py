
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#data_model = 'experiments_timespan_moodle.csv'
data_model = 'experiments_timespan_phpmyadmin.csv'

# read data from url as pandas dataframe
df_model = pd.read_csv(data_model)

#precision
sns.boxplot(y='precision', x='balancing',
                 data=df_model,
                 palette="Set2", #palette="colorblind", palette="Set2",["m", "g"]
                 hue="dataset"
                )
sns.despine(offset=10, trim=True)
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