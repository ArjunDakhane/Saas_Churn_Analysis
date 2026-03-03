SaaS Customer Churn Prediction System

This project is a business-oriented machine learning application built using Streamlit and XGBoost to predict customer churn probability. It provides risk segmentation, model evaluation, feature-level insights, and business recommendations for retention strategies.

The system simulates how a SaaS company could monitor churn risk and take proactive action.

⸻

Project Overview

Customer churn directly impacts revenue and growth in subscription-based businesses.

This application:
	•	Trains an XGBoost classifier on telecom churn data
	•	Handles preprocessing and encoding automatically
	•	Evaluates performance using ROC-AUC and confusion matrix
	•	Segments customers into High, Medium, and Low risk groups
	•	Provides feature contribution analysis for predictions
	•	Suggests actionable retention strategies

The model is cached and saved using joblib to avoid retraining on every run.

⸻

Features

Model Training
	•	Cleans and preprocesses dataset
	•	Converts numeric columns properly
	•	Encodes categorical features using LabelEncoder
	•	Handles class imbalance using scale_pos_weight
	•	Trains an XGBoost classifier
	•	Caches model with @st.cache_resource
	•	Saves trained model as model.pkl

⸻

Model Evaluation
	•	ROC-AUC score
	•	Confusion matrix (threshold = 0.5)

⸻

Risk Segmentation Dashboard

Customers are categorized based on predicted churn probability:
	•	Probability ≥ 0.7 → High Risk
	•	Probability between 0.4 and 0.69 → Medium Risk
	•	Probability < 0.4 → Low Risk

The dashboard displays:
	•	Count of customers in each risk segment
	•	Customer lists filtered by risk level

⸻

Individual Customer Prediction

The application allows manual feature input for a new customer and:
	•	Predicts churn probability
	•	Displays risk level
	•	Shows top 3 contributing features using XGBoost feature contributions
	•	Provides a business recommendation based on risk level

⸻

Tech Stack
	•	Python
	•	Streamlit
	•	Pandas
	•	NumPy
	•	Scikit-learn
	•	XGBoost
	•	Joblib

⸻

Project Structure

churn_project/
│
├── app.py
├── model.pkl (auto-generated)
│── Telco_customer_churn2.csv
└── README.md


⸻

Installation and Setup

1. Clone the Repository

git clone https://github.com/your-username/churn-project.git
cd churn-project

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux

3. Install Dependencies

If you have a requirements file:

pip install -r requirements.txt

Otherwise:

pip install streamlit pandas numpy scikit-learn xgboost joblib


⸻

Running the Application

streamlit run app.py

The application will open in your browser at:

http://localhost:8501


⸻

Model Details
	•	Algorithm: XGBoost Classifier
	•	Evaluation Metric: ROC-AUC
	•	Class imbalance handled using scale_pos_weight
	•	Feature contribution extracted using:

booster.predict(pred_contribs=True)

⸻

Business Value

This system can support:
	•	Customer Success teams in identifying churn-risk customers
	•	Sales teams in upselling low-risk customers
	•	Product teams in understanding churn-driving factors
	•	Management in monitoring overall churn health

⸻

Future Improvements
	•	SHAP-based visualization
	•	Model retraining option in UI
	•	FastAPI backend version
	•	AWS deployment
	•	Database integration
	•	Real-time prediction endpoint
	•	Automated email trigger for high-risk customers

⸻

Author
Arjun Dakhane
