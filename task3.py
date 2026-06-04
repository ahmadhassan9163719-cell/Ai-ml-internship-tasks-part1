import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.model_selection as ms
import sklearn.tree as tree

# Load the heart disease dataset from a highly reliable public repository
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
data = pd.read_csv(url)

# Clean the dataset by handling missing values
# This line drops any rows that contain missing or empty cells
data = data.dropna()

# Perform basic exploratory data analysis
# Print the shape and verify the column names
print("Dataset shape details:")
print(data.shape)
print("\nFirst few rows of the medical data:")
print(data.head())

# Split the dataset into features and our target variable
# The target column has 1 for heart disease risk and 0 for healthy
X = data.drop(columns=["target"])
y = data["target"]

# Split the data into 80 percent training and 20 percent testing sets
X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize a Decision Tree Classifier model
# We set max depth to keep the flowchart simple and prevent overfitting
model = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

# Train the classification model using our training data
model.fit(X_train, y_train)

# Make prediction guesses on the test data
predictions = model.predict(X_test)

# Calculate the classification accuracy metric
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"\nModel classification accuracy score: {accuracy:.2f}")

# Generate and display the confusion matrix data
confusion = metrics.confusion_matrix(y_test, predictions)
print("\nConfusion matrix layout:")
print(confusion)

# Plot the confusion matrix visually using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Heart Disease Prediction Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Calculate the data points for the receiver operating characteristic curve
# This uses the probability scores of the predictions
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities)
roc_auc = metrics.auc(fpr, tpr)

# Plot the receiver operating characteristic curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.title("Receiver Operating Characteristic Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Extract and display the feature importance attributes
# This shows which medical metrics mattered most to the model
importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\nImportant features ranking affecting prediction:")
print(importance)

# Plot the feature importances for a quick visual summary
plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x="Importance", y="Feature", palette="viridis")
plt.title("Important Features Affecting Heart Disease Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Medical Metric Column")
plt.show()