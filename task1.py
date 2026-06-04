
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Seaborn has the Iris dataset built right into it for practice
df = sns.load_dataset("iris")


print("1. Shape of dataset")
print(df.shape)  # Returns (rows, columns)
print("\n" + "=" * 40 + "\n")

print("2. Column names")
print(df.columns)
print("\n" + "=" * 40 + "\n")

print("3. First few rows")
print(df.head())
print("\n" + "=" * 40 + "\n")

print("4. Dataset info")
print(df.info())  # Shows data types and missing values
print("\n" + "=" * 40 + "\n")

print("5. Descriptive statistics")
print(df.describe())  # Shows mean, median, min, max, etc.
print("\n" + "=" * 40 + "\n")



#Plot 1: Scatter Plot 

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="petal_length", y="petal_width", hue="species", palette="deep"
)
plt.title("Scatter Plot: Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()

# Plot 2: Histogram
# We look at the distribution of Sepal Length for the flowers
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="sepal_length", kde=True, color="skyblue")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()

# Plot 3: Box Plot
# We check the distribution and look for outliers in Sepal Width across species
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="sepal_width", palette="pastel")
plt.title("Box Plot: Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()