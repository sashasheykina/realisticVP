from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

dp = pd.read_csv("experiments_without.csv")
dn = pd.read_csv("experiments_with.csv")
stats.probplot(dp['mcc'], dist="norm", plot=plt)
plt.title("Blood Pressure Before Q-Q Plot")
plt.savefig("BP_Before_QQ.png")

stats.probplot(dn['mcc'], dist="norm", plot=plt)
plt.title("Blood Pressure After Q-Q Plot")
plt.savefig("BP_After_QQ.png")

print(stats.wilcoxon(dp['mcc'], dn['mcc']))
