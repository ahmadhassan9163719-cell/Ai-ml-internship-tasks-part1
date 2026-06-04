import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as ms
import sklearn.ensemble as ens
import yfinance as yf

# Fetch historical data for Apple stock
# We fetch 2 years of daily data
ticker = "AAPL"
data = yf.download(ticker, start="2024-01-01", end="2026-01-01")

# Clean the data by removing any missing values
data = data.dropna()

# Define our features (inputs) and target (output)
# We use Open, High, Low, and Volume to predict the Close price
features = ["Open", "High", "Low", "Volume"]
X = data[features]

# We use squeeze to flatten the target data into one dimension
y = data["Close"].squeeze()

# Split the data into a training set and a testing set
# 80 percent of the data is used for training and 20 percent for testing
X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Initialize the Random Forest Regressor model
# This model creates multiple decision trees to make predictions
model = ens.RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Create a DataFrame to compare the actual values with predictions
comparison = pd.DataFrame(
    {"Actual Price": y_test, "Predicted Price": predictions}, index=y_test.index
)

# Print the first few rows of the comparison table
print("First few rows of actual versus predicted prices:")
print(comparison.head())

# Plot the actual prices versus the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(
    comparison.index,
    comparison["Actual Price"],
    label="Actual Price",
    color="blue",
)
plt.plot(
    comparison.index,
    comparison["Predicted Price"],
    label="Predicted Price",
    color="orange",
    linestyle="--",
)

# Add titles and labels to the plot
plt.title("Apple Stock Price Prediction: Actual versus Predicted")
plt.xlabel("Date")
plt.ylabel("Stock Price in USD")
plt.legend()
plt.show()