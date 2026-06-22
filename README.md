# AI/ML Engineering Internship Tasks

This repository contains my completed machine learning and deep learning tasks for the DevelopersHub Corporation internship.

---

## BASIC INTERNSHIP TASKS

### Task 1: Exploring and Visualizing a Simple Dataset
* **Objective:** Learn how to load, inspect, and visualize a data distribution.
* **Dataset Used:** Built-in Iris Dataset.
* **Key Results:** Created scatter plots, histograms, and box plots to see flower physical traits and spot outliers.

### Task 2: Predict Future Stock Prices (Short-Term)
* **Objective:** Predict the next day's closing stock price using time series historical values.
* **Dataset Used:** Apple (AAPL) data via yfinance.
* **Model Applied:** Random Forest Regressor.
* **Key Results:** Generated a graph comparing actual versus predicted values to track stock trends.

### Task 3: Heart Disease Prediction
* **Objective:** Build a medical classification system to determine heart disease risk.
* **Dataset Used:** UCI Heart Disease Dataset.
* **Model Applied:** Decision Tree Classifier.
* **Key Results:** Generated an accuracy score, evaluation confusion matrix, and ranked feature importance to see top health metrics.

---

## ADVANCED INTERNSHIP TASKS (Deadline: 30th June, 2026)

This section covers advanced machine learning, deep learning, and transformer implementations involving automated processing lines, sequence classification, and multimodal layouts.

### Advanced Task 1: News Topic Classifier Using BERT
* **Objective:** Fine-tune a deep learning transformer architecture to automatically categorize news headlines.
* **Approach:** Loaded raw text variables straight from public parquet shards to bypass package validation loops. Tokenized strings using a pre-trained `bert-base-uncased` language processor and executed optimization tuning via the Hugging Face `Trainer` utility. Absolute local paths were established to host a live text prediction application via a Gradio interface.
* **Key Results:** Built an active local web server allowing users to type input titles and receive accurate real-time classifications into World, Sports, Business, or Sci/Tech streams based on semantic intent.

### Advanced Task 2: End-to-End ML Pipeline with Scikit-Learn
* **Objective:** Construct an automated production line capable of cleaning unstructured, messy company data rows containing blanks and strings without manual script adjustments.
* **Approach:** Structured a cohesive Scikit-Learn `Pipeline` using a unified `ColumnTransformer`. The numerical track processes missing fields via median value imputation and applies standard scaling; the text categorical track maps empty elements to constant placeholders and encodes strings via One-Hot binary transformation. The cleaned records train a `Gradient Boosting Classifier`, which is compiled into a static `pipeline_model.pkl` binary artifact utilizing `joblib`.
* **Key Results:** Loaded the automated model artifact back into a dynamic web UI powered by `Streamlit` where end-users can adjust customer parameters on a dashboard and instantly receive underlying clean-up and churn risk metrics.

### Advanced Task 3: Multimodal Image-Text Matching & Zero-Shot Classification
* **Objective:** Connect entirely distinct visual and linguistic data streams simultaneously within a single computational framework.
* **Approach:** Utilized OpenAI's pre-trained `clip-vit-base-patch32` multimodal network architecture to project raw uploaded pixel arrays and arbitrary lists of comma-separated string labels into a single shared mathematical embedding space without downstream retraining.
* **Key Results:** Engineered an operational `Gradio` visual submission web interface where users upload a graphic, input arbitrary classification candidate labels, and instantly receive cross-modal semantic proximity scores displayed as a live correlation chart.
