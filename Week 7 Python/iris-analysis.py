 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

 Load the Iris dataset
def load_data()
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
    return df
except Exception as e:
    print(f"Error loading dataset: {e}")

 Display first few rows
def explore_data(df)
print("First 5 rows of the dataset:")
print(df.head()
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

 Statistical summary
def statistical_summary(df)
print("\nStatistical Summary:")
print(df.describe())

 Group by species and compute the mean
def gruop_by_species(df)
grouped = df.groupby('species').mean()
print("\nMean of features grouped by species:")
print(grouped)

Visualization Setup
def visualize_data(df)
sns.set(style="whitegrid")

 1. Line plot - Mean of each feature per species
gruopes=df.groupby('species').mean()
grouped.T.plot(kind='line', marker='o')
plt.title("Average Feature Values by Species")
plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.legend(title='Species')
plt.grid(True)
plt.tight_layout()
plt.show()

2. Bar chart - Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.tight_layout()
plt.show()

 3. Histogram - Distribution of sepal length
plt.figure(figsize=(6, 4))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

 4. Scatter plot - Sepal vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

Execution
if_name_="_main_"
df=load_data()
explore data(df)
statistical_summary(df)
group_by_species(df)
visualize_data(df)

 Observations
print("\nObservations:")
print("- Setosa species have significantly smaller petal lengths.")
print("- Sepal and petal lengths show a clear distinction between species.")
print("- Distributions and group means indicate strong separability among species.")