import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("output/merged_dataset.csv", sep=";")

print(df.info())
print(df.describe())

# Distribution de la gravité
sns.countplot(x=df["grav"])
plt.title("Distribution de la gravité")
plt.savefig("output/plots/distribution_gravite.png")
plt.close()
