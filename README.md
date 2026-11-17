# Credit Ledger – AI-Powered Credit Risk Prediction System

Credit Ledger is a machine learning-based credit risk assessment application that estimates the likelihood of a loan applicant defaulting on a loan. It uses applicant information, financial details, and credit history to generate a default probability and classify applications into **High Risk** or **Low Risk** categories.

The project combines an XGBoost classification model, a FastAPI backend, and an interactive web interface to demonstrate an end-to-end machine learning application.

## Live Demo

**Live Application:** [Credit Ledger – Credit Risk Assessment](https://cred-risk.onrender.com)

Explore the application by entering applicant details and viewing the predicted loan default probability and risk category.

## Problem Statement

Financial institutions need to assess the repayment risk associated with loan applications. Evaluating applicants using multiple financial and credit-related factors can be challenging, especially when dealing with large datasets.

Credit Ledger aims to simplify this process by using machine learning to estimate loan default risk based on applicant characteristics, loan details, and credit history.

The system provides a probability-based risk assessment to support informed lending decisions. It is an educational decision-support project, not a replacement for professional credit evaluation.

## Goals

- Develop an end-to-end machine learning application for credit risk assessment.
- Predict the probability of loan default using applicant and loan-related information.
- Apply a custom decision threshold to categorize applications into risk levels.
- Improve model interpretability using SHAP.
- Explore probability calibration for more reliable risk estimates.
- Deploy the trained model through a REST API and interactive web interface.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and preprocessing |
| Scikit-learn | Preprocessing, evaluation and ML utilities |
| XGBoost | Loan default classification |
| SHAP | Model explainability |
| FastAPI | Backend API development |
| Pydantic | Input validation |
| HTML | Web page structure |
| CSS | User interface styling |
| JavaScript | Frontend interactions and API integration |
| Joblib | Model serialization and loading |
| Render | Cloud deployment |

## Key Features

### 1. Loan Default Prediction
Estimates the probability that an applicant may default based on financial and credit-related attributes.

### 2. Risk Classification
Classifies applications into:
- **Low Risk:** Predicted probability is below the selected threshold.
- **High Risk:** Predicted probability meets or exceeds the selected threshold.

### 3. Custom Decision Threshold
Uses a custom threshold of **71.7%** instead of relying on the default 50% classification threshold.

### 4. Probability Calibration
Calibration helps make predicted probabilities better reflect the actual frequency of outcomes. It is useful when probability estimates are important for risk assessment.

### 5. SHAP-Based Explainability
SHAP (SHapley Additive exPlanations) helps interpret model predictions by showing how input features contribute to the predicted outcome.

It can provide:
- **Global Interpretation:** Which features influence predictions across the dataset.
- **Local Interpretation:** How individual features contribute to one applicant's prediction.

### 6. Interactive Web Interface
Provides a form to enter applicant details and view the predicted default probability and risk category.

### 7. Automatic Loan-to-Income Calculation
Automatically calculates the loan amount as a proportion of annual income, helping assess the relative loan burden.

### 8. REST API Integration
Uses FastAPI to connect the trained machine learning model with the frontend and serve predictions.

## Input Features and Interpretation

| Feature | Description |
|---|---|
| Person Age | Applicant's age in years. |
| Person Income | Applicant's annual income. |
| Home Ownership | Housing status: RENT, MORTGAGE, OWN or OTHER. |
| Employment Length | Number of years employed. |
| Loan Intent | Purpose of the loan. |
| Loan Grade | Existing lender-assigned loan risk grade (A–G). |
| Loan Amount | Total amount requested by the applicant. |
| Interest Rate | Interest rate associated with the loan. |
| Loan Percent Income | Loan amount divided by annual income. |
| Previous Default | Whether a previous default is recorded (Y/N). |
| Credit History Length | Number of years of recorded credit history. |

## Model Output

The application provides:

- **Default Probability:** Estimated probability of loan default.
- **Decision Threshold:** Custom classification threshold of 71.7%.
- **Risk Category:** High Risk or Low Risk.

The threshold determines the final classification; it does not change the underlying predicted probability.

## Model Interpretation

The project uses SHAP to understand the contribution of different input features to model predictions.

For example:
- Higher loan-to-income ratios may push predictions towards higher risk.
- A previous default record may increase estimated risk.
- A stronger credit grade may be associated with lower estimated risk.

These are general interpretations, not guaranteed effects for every applicant. SHAP explains model behavior, not cause-and-effect relationships.


## Disclaimer

Credit Ledger is an academic and demonstration project developed for learning and showcasing machine learning deployment and model interpretation.

Predictions are generated from patterns learned from the training dataset and should not be treated as actual financial advice, official credit scores, loan approval decisions, or guarantees of future repayment behavior.

The application does not replace professional financial assessment or the lending policies of financial institutions.


