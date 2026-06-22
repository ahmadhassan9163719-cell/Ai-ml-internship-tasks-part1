import joblib
import numpy as np
import pandas as pd
import sklearn.compose as compose
import sklearn.ensemble as ens
import sklearn.impute as impute
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocess

# Generate a synthetic messy dataset to simulate real world company records
np.random.seed(42)
n_samples = 1000

# Create columns containing numbers and text categories with missing values
raw_data = {
    "Age": np.random.choice([25, 35, 45, np.nan, 55], size=n_samples),
    "Monthly_Spend": np.random.choice(
        [50.0, 80.0, np.nan, 120.0, 150.0], size=n_samples
    ),
    "Subscription_Type": np.random.choice(
        ["Basic", "Standard", "Premium", np.nan], size=n_samples
    ),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
}

df = pd.DataFrame(raw_data)

# Separate inputs from our target label column
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Define which columns are numbers and which are text categories
numeric_features = ["Age", "Monthly_Spend"]
categorical_features = ["Subscription_Type", "Gender"]

# Step 1: Create the sub-assembly line for numerical columns
# It fills blank spots with the median and scales values for the model
numeric_transformer = pipeline.Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer(strategy="median")),
        ("scaler", preprocess.StandardScaler()),
    ]
)

# Step 2: Create the sub-assembly line for categorical text columns
# It fills blanks with a constant placeholder and applies One-Hot Encoding
categorical_transformer = pipeline.Pipeline(
    steps=[
        (
            "imputer",
            impute.SimpleImputer(strategy="constant", fill_value="missing"),
        ),
        ("onehot", preprocess.OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Step 3: Combine both processing paths into a unified ColumnTransformer
preprocessor = compose.ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Step 4: Combine the preprocessor and the Machine Learning model into a single pipeline
# We use Gradient Boosting Classifier as requested by the advanced guidelines
full_pipeline = pipeline.Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", ens.GradientBoostingClassifier(random_state=42)),
    ]
)

# Train the entire end-to-end pipeline at once
print("Training the end-to-end pipeline model...")
full_pipeline.fit(X, y)

# Save the complete pipeline object to a file on your hard drive
model_filename = "pipeline_model.pkl"
joblib.dump(full_pipeline, model_filename)

print(f"Pipeline successfully built and saved to {model_filename}")