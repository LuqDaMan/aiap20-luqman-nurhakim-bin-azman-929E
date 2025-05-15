# Bank Term Deposit Subscription Prediction Pipeline

**Author:** Luqman Nurhakim B Azman
**Email:** lluqmannurhakim@gmail.com
**Project Version:** 1.1 (Includes ML Pipeline and Local Deployment with UI)
**Date:** 2025-05-12

## 1. Overview of Submitted Folder and Project

This project implements an end-to-end Machine Learning Pipeline (MLP) to predict the likelihood of a bank client subscribing to a term deposit. The prediction is based on client demographic data, financial status, and past campaign interaction data. The primary goal is to enable the bank to optimize marketing strategies by targeting high-propensity clients.

The project encompasses:
1.  **Exploratory Data Analysis (EDA):** An in-depth analysis of the client dataset to uncover patterns, identify data quality issues, and derive insights for feature engineering and model building. Full details are in `notebooks/eda.ipynb`.
2.  **End-to-End Machine Learning Pipeline:** A configurable and modular pipeline for data ingestion, preprocessing, feature engineering, training and evaluating multiple classification models (Logistic Regression, Random Forest, Gradient Boosting), and logging experiments with MLflow.
3.  **Local Deployment & Interface:** Local deployment of the trained models using MLflow model serving, a FastAPI backend acting as a proxy, and an interactive Streamlit frontend for predictions.

The repository is structured to separate concerns, with data, configuration, source code, notebooks, scripts, and logs organized into distinct directories.

### 1.1. Folder Structure

```
aiap20-luqman-nurhakim-bin-azman-929E/
├── config/
│   ├── pipeline_config.yaml       # Configuration for the ML pipeline
│   └── deploy_config.yaml         # Configuration for deployment (API, MLflow servers, Streamlit)
├── data/
│   └── bmarket.db                 # SQLite database (Source of data)
├── logs/
│   ├── pipeline/                  # Logs from ML pipeline runs (run.sh)
│   ├── api/                       # Logs from FastAPI server
│   ├── mlflow_model_servers/      # Logs from MLflow model servers
│   └── streamlit/                 # Logs from Streamlit application
├── mlruns/                        # MLflow experiment tracking data
├── artifacts/                     # Saved artifacts like the preprocessor (preprocessor.joblib)
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis notebook
│   └── scratch.ipynb              # Scratchpad for experimentation
├── scripts/
│   └── update_model_uris.py       # Script to update model URIs in deploy_config.yaml
├── src/
│   ├── init.py
│   ├── api/
│   │   ├── init.py
│   │   ├── main.py                # FastAPI application
│   │   └── schemas.py             # Pydantic schemas for API
│   ├── app/
│   │   ├── init.py
│   │   └── streamlit_main.py      # Streamlit frontend application
│   ├── utils/
│   │   ├── init.py
│   │   ├── api_utils.py           # Utility functions for the API
│   │   ├── pipe_utils.py          # Utility functions for the ML pipeline (logging, config)
│   │   └── stream_utils.py        # Utility functions for the Streamlit app
│   ├── data_ingestion.py          # Data loading module
│   ├── preprocessing.py           # Data cleaning and initial preprocessing module
│   ├── feat_engin.py              # Feature engineering (encoding, scaling) and data splitting module
│   ├── model_trainer.py           # Model training and tuning module
│   ├── model_evaluator.py         # Model evaluation module
│   └── pipeline.py                # Main ML pipeline script (orchestrator)
├── tests/
│   └── deploy/
│       └── pyfunc_test.py         # Example test script for MLflow PyFunc models
│   └── pipeline/
│       
├── .gitignore
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run.sh                         # Script to run the ML pipeline
└── run_app.sh                     # Script to run the deployed application (servers & UI)
```
## 2. Instructions for Execution and Configuration

### 2.1. Prerequisites
* Python 3.9+
* Access to a terminal or command prompt.
* Git for cloning the repository.

### 2.2. Setup
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/LuqDaMan/aiap20-luqman-nurhakim-bin-azman-929E.git
    cd aiap20-luqman-nurhakim-bin-azman-929E
    ```

2.  **System Prerequisites:**
    * **Homebrew (macOS):** Ensure Homebrew is installed. If not, follow instructions at [https://brew.sh/](https://brew.sh/).
    * **jq:** This project uses `yq` for parsing YAML configuration files, and `yq` in turn requires `jq`. Install `jq` using Homebrew:
        ```bash
        brew install jq
        ```
    * **Python 3:** Ensure Python 3 (e.g., 3.8+) is installed.

3.  **Environment Setup & Dependencies:**
    The `run.sh` and `run_app.sh` scripts handle Python virtual environment creation and dependency installation from `requirements.txt`.
    To set up manually:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Start MLflow Tracking Server (Optional, if not using `run.sh`):**
    If you wish to run the MLflow UI independently (note: `run.sh` starts its own MLflow server instance):
    ```bash
    mlflow ui --host 127.0.0.1
    ```
    Then access the MLflow dashboard at http://127.0.0.1:5000 in your browser.
   

### 2.3. Running the ML Pipeline
This will ingest data, preprocess it, engineer features, train models, evaluate them, and log to MLflow.
```bash
bash run.sh
```

**Modifying Pipeline Parameters:**
Edit the `config/pipeline_config.yaml` file to change parameters like:
* `random_seed`
* `logging_settings`
* `data_source` (DB path, table)
* `output_paths`
* Specific feature processing parameters (e.g., `age_processing`, `loan_cols_missing_handling`)
* `education_level_order` for ordinal encoding
* `train_test_split` ratios
* `imbalance_handling` method (e.g., "SMOTE")
* `models_to_train`
* Hyperparameter tuning grids and CV folds
* `evaluation_metrics`

### 2.4. Running the Deployed Application
This will start MLflow servers for the trained models, a FastAPI backend, and a Streamlit UI.

**Ensure ML Pipeline has run:** The `run.sh` script must have been executed successfully at least once to train and log models to MLflow.

**Update Model URIs (Important):**
To ensure the deployment uses the latest trained models from your MLflow runs, execute:
```bash
# Ensure your virtual environment is activated: source .venv/bin/activate
python3 scripts/update_model_uris.py
```
This script updates `config/deploy_config.yaml` with the latest MLflow run IDs for the Logistic Regression, Random Forest, and Gradient Boosting models.

**Launch the Application:**
```bash
bash run_app.sh
```
Or, to specify a custom deployment configuration file:
```bash
bash run_app.sh path/to/your_custom_deploy_config.yaml
```
This script will launch:
* MLflow Tracking Server (Default: `http://localhost:5001`)
* MLflow Model Servers for each model (Ports defined in `deploy_config.yaml`, e.g., 5002, 5003, 5004)
* FastAPI Backend Server (Default: `http://localhost:8000`)
* Streamlit Frontend UI (Default: `http://localhost:8501`)

**Modifying Deployment Parameters:**
Edit the `config/deploy_config.yaml` file to change parameters like:
* API settings (FastAPI host, port)
* MLflow tracking URI for deployment scripts
* `mlflow_model_servers` configurations (host, port, model URI for each model)
* `streamlit_app` settings (title, API URL, input form fields)
* Logging settings for API, model servers, and Streamlit.

## 3. Logical Steps/Flow of the Pipeline
The project involves two main execution flows: the ML pipeline (`run.sh`) and the deployed application (`run_app.sh`).

### 3.1. ML Pipeline (`run.sh`) Logical Flow
The `run.sh` script orchestrates the training pipeline defined in `src/pipeline.py`. The logical flow is as follows:
```text
[SQLite DB (data/bmarket.db)]
       |
       V
[1. Data Ingestion (src/data_ingestion.py)]
   - Loads raw data from the 'bank_marketing' table.
       |
       V
[2. Data Cleaning & Initial Preprocessing (src/preprocessing.py)]
   - Standardizes column names.
   - Handles anomalies (e.g., 'age').
   - Standardizes categorical strings (e.g., 'occupation', 'contact_method').
   - Manages missing/unknown values in specified columns (e.g., 'housing_loan', 'personal_loan').
   - Processes 'campaign_calls' (negatives) and 'previous_contact_days' (999).
   - Encodes the target variable 'subscription_status'.
   - Drops 'client_id'.
   (Configuration driven by `pipeline_config.yaml`)
       |
       V
[3. Feature Engineering & Data Splitting (src/feat_engin.py)]
   - Creates new binary features ('previously_contacted', 'cc_had_negative_adjustment').
   - Applies transformations:
     - Log transform for skewed numerical features ('campaign_calls', 'previous_contact_days').
     - Standard scaling for all numerical features.
     - One-Hot Encoding for nominal categorical features.
     - Ordinal Encoding for 'education_level'.
   - Uses `sklearn.compose.ColumnTransformer` to apply these transformations systematically.
   - Splits data into training and testing sets (e.g., 80/20 split, stratified).
   - Fits the `ColumnTransformer` (preprocessor) on the training data and saves it to `artifacts/preprocessor.joblib`.
       |
       V
[4. Model Training & Tuning (src/model_trainer.py)]
   - Trains Logistic Regression, Random Forest, and Gradient Boosting models.
   - Performs hyperparameter tuning using GridSearchCV with cross-validation.
   - Handles class imbalance using SMOTE (on training folds within CV).
   - Logs trained models, parameters, and cross-validation metrics to MLflow.
       |
       V
[5. Model Evaluation (src/model_evaluator.py)]
   - Evaluates the best tuned models (from GridSearchCV) on the unseen test set.
   - Calculates metrics: F1-score (yes class), AUC-ROC, AUPRC, Precision, Recall.
   - Generates and logs classification reports, confusion matrices, ROC curves, PR curves, and SHAP plots to MLflow for each model.
       |
       V
[MLflow Tracking (mlruns/)]
   - Stores all experiment runs, parameters, metrics, saved preprocessor, trained models, and plots.
```
This pipeline is designed with modularity and configurability in mind, allowing for easy modifications and experimentation. The `pipeline_config.yaml` file centralizes all controllable parameters.

### 3.2. Deployed Application (`run_app.sh`) Logical Flow
The `run_app.sh` script starts the services required for the interactive prediction application:

1.  **MLflow Tracking Server:** Serves the MLflow UI and provides access to logged experiments and models.
2.  **MLflow Model Servers:** Three separate instances of `mlflow models serve` are started, one for each of the best Logistic Regression, Random Forest, and Gradient Boosting models retrieved from MLflow (based on URIs in `deploy_config.yaml`). Each serves predictions on a specific port.
3.  **FastAPI Backend (`src/api/main.py`):**
    *   Acts as a proxy to the MLflow model servers.
    *   Exposes a `/predict/` endpoint that accepts raw client data and a selected model name.
    *   Loads the saved `preprocessor.joblib` from the ML pipeline run.
    *   Preprocesses the incoming raw data using this preprocessor.
    *   Forwards the preprocessed data to the appropriate MLflow model server based on the selected model.
    *   Returns the prediction (class and probability) from the model server.
    *   Also exposes a `/health` endpoint.
4.  **Streamlit Frontend (`src/app/streamlit_main.py`):**
    *   Provides a user interface with input fields for client data.
    *   Allows users to select one of the three deployed models.
    *   Sends the raw client data and model choice to the FastAPI `/predict/` endpoint.
    *   Displays the prediction received from FastAPI.

This setup allows for interactive predictions without needing to rerun the entire training pipeline.

## 4. EDA Summary and Pipeline Choices
Exploratory Data Analysis (EDA) was conducted on the `bank_marketing` dataset. Full details and visualizations are in `notebooks/eda.ipynb`. The insights from EDA directly informed the design of the preprocessing and feature engineering stages in the ML pipeline.

**Key EDA Findings & Resulting Pipeline Choices:**

*   **Data Quality Issues & Handling:**
    *   Anomalous `age` values (e.g., "150 years"): Cleaned by removing " years", converting to numeric, and imputing values > 100 with the median of valid ages. This results in the `cleaned_age` feature.
    *   Inconsistent categorical strings (e.g., "admin." vs "admin", "Cell" vs "cellular" in `contact_method`): Standardized to lowercase, stripped whitespace, and specific consolidations like "cell" to "cellular" were applied.
    *   Missing values (NaNs) and "unknown" strings:
        *   `housing_loan`, `personal_loan`: NaNs and "unknown" strings mapped to a distinct 'unknown' category.
        *   `occupation`, `marital_status`, `education_level`, `credit_default`: "unknown" strings standardized and treated as a distinct category during encoding.
    *   Illogical `campaign_calls` values (negatives): Absolute values taken, and a binary indicator `cc_had_negative_adjustment` created.
*   **Feature Engineering Based on EDA:**
    *   `previous_contact_days` = 999: This special value, indicating no previous contact, was used to create a binary `previously_contacted` feature (True if not 999 and not NaN, False otherwise). The `previous_contact_days` column itself was then modified to set these 999 values to 0. This transformation was crucial as EDA (e.g., Mutual Information scores) highlighted the importance of previous contact.
*   **Target Variable Imbalance:**
    *   `subscription_status` showed significant imbalance (~11% 'yes'). This led to the choice of SMOTE for handling imbalance during model training and the selection of F1-score for the 'yes' class as a primary evaluation metric.
*   **Feature Distributions & Transformations:**
    *   Numerical features like `campaign_calls` (absolute) and `previous_contact_days` (for contacted clients) were found to be right-skewed in EDA. This justified applying a log transformation (`np.log1p`) before scaling to make their distributions more symmetric for modeling.
    *   All numerical features were standardized using `StandardScaler` after any transformations.
*   **Encoding Categorical Features:**
    *   Nominal features (e.g., `occupation`, `marital_status`) are One-Hot Encoded.
    *   `education_level` is Ordinal Encoded based on a predefined order, reflecting its inherent hierarchy observed in EDA.

These choices ensure data quality, create meaningful features, and prepare the data appropriately for the selected machine learning models.

## 5. Feature Processing Summary
The following table summarizes how key features from the dataset are processed throughout the pipeline, based on `pipeline_config.yaml`:

| Feature Group / Specific Feature        | Initial Cleaning & Standardization (in `preprocessing.py`)                                     | Transformation & Encoding (in `feat_engin.py` via `ColumnTransformer`)                          | Justification                                              |
|-------------------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| Column Names                              | Standardized to `lowercase_with_underscores`.                                                  | N/A                                                                                             | Ensures consistent naming conventions for easier access.     |
| `client_id`                               | Dropped.                                                                                       | N/A                                                                                             | Not relevant for model training; an identifier.              |
| `age`                                     | " years" removed, converted to numeric. Values > 100 or NaN imputed with median of valid ages. Result stored in `cleaned_age`. Original `age` dropped. | `cleaned_age` is treated as a numerical feature: Scaled with `StandardScaler`.                | Handles anomalous age entries and prepares for numerical modeling. |
| `campaign_calls`                          | Negative values converted to absolute. Binary `cc_had_negative_adjustment` created. NaNs (if any after abs) imputed with median. | `campaign_calls` (abs value): Log-transformed (`np.log1p`), then `StandardScaler`. `cc_had_negative_adjustment`: Used as is (binary 0/1). | Corrects illogical negative values, captures adjustment information, and normalizes distribution for modeling. |
| `previous_contact_days`                   | Binary `previously_contacted` created (True if not 999 & not NaN). Original column: 999s and NaNs replaced with 0. | `previous_contact_days` (modified): Log-transformed (`np.log1p`) for non-zero values, then `StandardScaler`. `previously_contacted`: Used as is (binary 0/1). | Creates a meaningful binary indicator for prior contact and normalizes the skewed numerical feature. |
| `occupation`, `marital_status`, `credit_default`, `contact_method` | Standardized to lowercase, stripped whitespace. `contact_method` values consolidated (e.g., "cell" to "cellular"). "unknown" treated as valid string. | One-Hot Encoded.                                                                              | Standardizes categorical inputs and converts them to a numerical format suitable for modeling. |
| `education_level`                         | Standardized to lowercase, stripped whitespace. "unknown" treated as valid string.             | Ordinal Encoded based on `education_level_order` in config.                                       | Preserves the inherent order of education levels during numerical conversion. |
| `housing_loan`, `personal_loan`           | NaNs and "unknown" strings mapped to a new distinct category (e.g., "unknown" as per `pipeline_config.yaml`). | One-Hot Encoded.                                                                              | Handles missing/unknown values by creating a distinct category and prepares for numerical modeling. |
| Other Numerical Features (if any not explicitly listed above but identified as numerical in config) | Assumed clean or handled by general rules.                                                       | `StandardScaler` applied.                                                                     | Standardizes numerical features to have zero mean and unit variance. |
| Target Variable (`subscription_status`)   | Encoded to 0 ('no') and 1 ('yes') based on `target_variable_encoding` map.                       | N/A (Target variable)                                                                           | Converts the categorical target into a binary numerical format for classification. |

This systematic processing, driven by EDA and configuration, ensures that the data fed into the models is clean, consistent, and in an optimal format for learning.

## 6. Choice of Models and Justification
The pipeline trains and evaluates three distinct classification models:

*   **Logistic Regression:**
    *   **Choice:** Chosen as a robust and interpretable baseline model. It provides a good starting point for understanding feature influences through its coefficients (after appropriate scaling).
    *   **Justification:** It's computationally efficient, less prone to overfitting on smaller datasets compared to more complex models, and offers insights into linear relationships between features and the target.
*   **Random Forest Classifier:**
    *   **Choice:** An ensemble learning method based on bagging decision trees.
    *   **Justification:** Random Forests are generally high-performing, robust to outliers and non-linear data, and can handle a large number of features. They provide feature importance measures, which can be valuable for understanding predictors. Their ensemble nature helps reduce overfitting compared to a single decision tree.
*   **Gradient Boosting Classifier:**
    *   **Choice:** An ensemble learning method based on boosting, building trees sequentially where each tree corrects the errors of its predecessor.
    *   **Justification:** Gradient Boosting machines are often among the top-performing models for tabular data. They can capture complex non-linear relationships and interactions between features. While potentially more prone to overfitting than Random Forests if not tuned carefully, their sequential learning process often leads to superior predictive accuracy.

These three models provide a good spectrum from simple/interpretable (Logistic Regression) to powerful ensembles (Random Forest, Gradient Boosting), allowing for a comprehensive evaluation of different modeling approaches for this binary classification task. All models are tuned using `GridSearchCV` to find optimal hyperparameters.

## 7. Model Evaluation and Metrics
Model evaluation is critical, especially given the business goal of identifying clients likely to subscribe ('yes' class) and the inherent class imbalance in the dataset (~11% 'yes').

**Evaluation Metrics Used:**

*   **F1-score (for 'yes' class):**
    *   **Explanation:** This is the harmonic mean of Precision and Recall for the positive class ('yes'). It's the primary metric for model selection because it provides a balance between correctly identifying subscribers (Recall) and ensuring that those predicted as subscribers are indeed subscribers (Precision). This is crucial when the cost of missing a potential subscriber (false negative) and the cost of incorrectly targeting a non-subscriber (false positive) both need to be considered.
    *   **Relevance:** Especially important for imbalanced datasets where accuracy can be misleading.
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**
    *   **Explanation:** Measures the model's ability to distinguish between the positive ('yes') and negative ('no') classes across all possible classification thresholds. An AUC of 1.0 indicates a perfect classifier, while 0.5 suggests a random classifier.
    *   **Relevance:** Provides a threshold-independent measure of separability.
*   **AUPRC (Area Under the Precision-Recall Curve):**
    *   **Explanation:** Summarizes the trade-off between Precision and Recall for the positive class across different thresholds.
    *   **Relevance:** Often more informative than AUC-ROC for highly imbalanced datasets because it focuses on the performance of the minority (positive) class. A higher AUPRC indicates better performance in identifying true subscribers while maintaining precision.
*   **Recall (Sensitivity or True Positive Rate for 'yes' class):**
    *   **Explanation:** Measures the proportion of actual positive instances (actual subscribers) that were correctly identified by the model. (TP / (TP + FN)).
    *   **Relevance:** High recall is important if the priority is to minimize missed opportunities (false negatives), i.e., capture as many potential subscribers as possible.
*   **Precision (Positive Predictive Value for 'yes' class):**
    *   **Explanation:** Measures the proportion of instances predicted as positive (predicted subscribers) that were actually positive. (TP / (TP + FP)).
    *   **Relevance:** High precision is important if the priority is to minimize wasted marketing efforts (false positives), i.e., ensure that targeted clients are indeed likely to subscribe.
*   **Classification Report:**
    *   **Explanation:** Provides a textual summary of precision, recall, and F1-score for each class, as well as overall accuracy and macro/weighted averages.
*   **Confusion Matrix:**
    *   **Explanation:** A table showing the counts of True Positives, True Negatives, False Positives, and False Negatives.
    *   **Relevance:** Gives a detailed breakdown of prediction performance and error types.

**Evaluation Process:**

1.  **Cross-Validation (CV):** During hyperparameter tuning (`GridSearchCV`), models are evaluated using stratified k-fold cross-validation (typically 5 folds as per `pipeline_config.yaml`) on the training data. This provides a robust estimate of performance and helps select the best hyperparameters.
2.  **Hold-out Test Set:** The final selected model (after tuning) for each algorithm type is then evaluated on a completely unseen test set to provide an unbiased estimate of its generalization performance.
3.  **MLflow Logging:** All these metrics, along with visualizations like ROC curves, PR curves, and SHAP plots (for feature importance/interpretability), are logged to MLflow for each model and run, allowing for easy comparison and tracking.

The choice of the "best" overall model is guided by the `primary_evaluation_metric` defined in `pipeline_config.yaml` (defaulting to `f1_score_yes_class`), but other metrics are considered to get a holistic view of performance.

## 8. Deployment Considerations
The current project includes local deployment using MLflow model serving, FastAPI, and a Streamlit UI. For a production environment, several other considerations would be crucial:

*   **Scalability & Performance:**
    *   The current MLflow and FastAPI setup (running via `run_app.sh`) is suitable for local demonstration but would need to be deployed on more robust infrastructure (e.g., Kubernetes, cloud-based managed services like AWS SageMaker, Google Vertex AI) to handle concurrent requests and ensure high availability.
    *   Load balancing would be necessary for multiple instances of the API.
    *   Asynchronous request handling in FastAPI (already used by default for path operations) can help, but underlying model servers also need to be scalable.
*   **Monitoring:**
    *   **Model Performance:** Tracking key metrics (F1, AUC-ROC, etc.) on live predictions to detect degradation over time.
    *   **Data Drift:** Monitoring input data distributions to detect significant changes from the training data, which could invalidate the model.
    *   **Concept Drift:** Monitoring changes in the relationship between features and the target variable.
    *   **System Health:** Logging, error rates, latency, and resource utilization of the deployed services. Tools like Prometheus, Grafana, or cloud-specific monitoring services would be used.
*   **Security:**
    *   **API Authentication/Authorization:** Implementing secure access to the prediction API (e.g., API keys, OAuth2).
    *   **Input Validation:** Robust validation of incoming data to prevent errors and potential security vulnerabilities.
    *   **Network Security:** Configuring firewalls and private networks.
*   **CI/CD (Continuous Integration/Continuous Deployment):**
    *   Automating the process of testing, building, and deploying new model versions or application updates.
    *   Tools like GitHub Actions, Jenkins, GitLab CI would be used. This would include automated testing of the pipeline and API.
*   **Model Versioning & Management:**
    *   MLflow already provides strong model versioning capabilities. A production setup would rigorously use the MLflow Model Registry to manage model stages (e.g., Staging, Production).
    *   Strategies for rolling out new model versions (e.g., canary releases, A/B testing) would be needed.
*   **Cost Optimization:**
    *   If deploying to the cloud, selecting appropriate instance types, auto-scaling policies, and serverless options to manage costs effectively.
*   **Containerization:**
    *   Using Docker to package the FastAPI application, MLflow model servers (if not using managed MLflow), and Streamlit app for consistent environments across development, testing, and production. This simplifies dependency management and deployment.
*   **Reproducibility & Governance:**
    *   Ensuring that every prediction can be traced back to the exact model version and data used.
    *   Maintaining audit trails for model training and deployment activities.
*   **Error Handling & Resilience:**
    *   More sophisticated error handling in the API and model servers, with fallback strategies or circuit breakers if necessary.

These considerations represent the shift from a local development/demonstration setup to a production-ready machine learning system.

## 9. Technology Stack Summary
*   **Core Language:** Python 3.x
*   **Data Handling:** Pandas, NumPy, SQLAlchemy (for SQLite)
*   **Machine Learning:** Scikit-learn, imbalanced-learn
*   **MLOps/Experiment Tracking:** MLflow
*   **API Development:** FastAPI, Uvicorn
*   **Frontend UI:** Streamlit
*   **Configuration:** PyYAML
*   **EDA & Notebooks:** Jupyter, Plotly, Matplotlib, Seaborn
*   **Logging:** Python's built-in `logging` module.
*   **Shell Scripting:** Bash.

## 10. Potential Next Steps / Future Work
Beyond the current scope and the deployment considerations listed above:

*   **Advanced Feature Engineering:** Explore interaction terms, polynomial features, or automated feature discovery.
*   **Broader Model Exploration:** Experiment with other algorithms like XGBoost, LightGBM, or simple Neural Networks.
*   **Automated Retraining Strategy:** Develop criteria and automation for when and how to retrain models based on performance degradation or new data availability.
*   **Enhanced UI/UX:** Add more visualizations to the Streamlit app, such as explanations for individual predictions (e.g., using SHAP waterfall plots if feasible).
*   **Comprehensive Data Validation:** Integrate libraries like `pydantic` more consistently for robust data validation throughout the pipeline and for incoming API requests.
*   **A/B Testing Framework:** Implement infrastructure for A/B testing different models in a live environment.


