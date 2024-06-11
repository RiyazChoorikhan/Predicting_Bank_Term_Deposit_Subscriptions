# Predicting_Bank_Term_Deposit_Subscriptions
Project Overview
This project aims to predict whether a client will subscribe to a term deposit based on various attributes collected during telephonic marketing campaigns conducted by a Portuguese banking institution. The main objectives are to preprocess the data, perform exploratory data analysis (EDA), engineer relevant features, build and evaluate predictive models, and deploy the final model. The significance of this project lies in improving the efficiency of marketing campaigns by targeting clients who are more likely to subscribe to term deposits, thereby optimizing resources and increasing revenue for the bank.

Data Sources and Descriptions
Banking Marketing Dataset from Kaggle

Origin: The dataset is publicly available on Kaggle and was collected from telephonic marketing campaigns conducted by a Portuguese banking institution between May 2008 and November 2010.
Description: The dataset contains detailed information on the bank's marketing campaigns, including client demographics, previous interactions, and campaign-specific details.
Files:
train.csv: 45,211 rows and 18 columns
test.csv: 4,521 rows and 18 columns
User-Provided Dataset

Description: This dataset contains similar information as the Kaggle dataset and is used to supplement the analysis.
File: test.csv
Instructions for Using the Code
Setup Instructions

Clone the GitHub repository to your local machine:
git clone https://github.com/RiyazChoorikhan/Predicting_Bank_Term_Deposit_Subscriptions/blob/main/Term_Dep_Subcribers_v1_20240609.py)
cd project-repo
Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Running the Scripts/Notebooks

Data Preprocessing: Run the preprocessing script to clean and prepare the data:
python scripts/preprocess_data.py
Exploratory Data Analysis: Open and run the Jupyter notebook notebooks/eda.ipynb to perform EDA.
Model Training: Execute the model training script:

python scripts/train_model.py
Model Evaluation: Use the evaluation script to assess model performance:

python scripts/evaluate_model.py
Contact Information
For questions or support, please contact the project maintainer:

Name: [Your Name]
Email: [your.email@example.com]
GitHub: https://github.com/username
Data Structure and Purpose
age: Client's age (numeric)
job: Type of job (categorical: "admin.","management","blue-collar","technician","services", etc.)
marital: Marital status (categorical: "married","divorced","single")
education: Education level (categorical: "unknown","secondary","primary","tertiary")
default: Has credit in default? (binary: "yes","no")
balance: Average yearly balance, in euros (numeric)
housing: Has housing loan? (binary: "yes","no")
loan: Has personal loan? (binary: "yes","no")
contact: Contact communication type (categorical: "unknown","telephone","cellular")
day: Last contact day of the month (numeric)
month: Last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")
duration: Last contact duration, in seconds (numeric)
campaign: Number of contacts during this campaign (numeric)
pdays: Days since the client was last contacted from a previous campaign (numeric, -1 means not previously contacted)
previous: Number of contacts before this campaign (numeric)
poutcome: Outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
y: Has the client subscribed a term deposit? (binary: "yes","no")
Usage and Preprocessing Steps
Loading the Data

Load the datasets using pandas:
python
Copy code
import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
Handling Missing Values

Check for missing values and handle them appropriately:
python
Copy code
train.isnull().sum()
test.isnull().sum()
Encoding Categorical Variables

Use label encoding or one-hot encoding for categorical variables:
python
Copy code
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(train[['job', 'marital', 'education', 'contact', 'month', 'poutcome']])
Normalization/Standardization

Normalize or standardize numeric features:
python
Copy code
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(train[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])
Splitting Data

Split the training data into training and validation sets:
python
Copy code
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train.drop('y', axis=1), train['y'], test_size=0.2, random_state=42)
Model Training and Evaluation

Train the model using the training set and evaluate it on the validation set:
python
Copy code
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_val)
By following these steps, users can preprocess the datasets and prepare them for analysis, ensuring the project setup is replicable and the data is ready for predictive modeling.

