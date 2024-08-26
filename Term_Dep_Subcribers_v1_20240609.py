import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


from google.colab import drive
drive.mount('/content/drive')

"""# Load datasets"""

train_df = pd.read_csv('/content/drive/MyDrive/SEM C/train.csv' , sep = ';')
test_df = pd.read_csv('/content/drive/MyDrive/SEM C/test.csv' , sep = ';')

"""# Initial Data Exploration"""

train_df.shape

train_df.head(5)

df = train_df

df.columns

df.nunique()

df.isnull().sum()

# Describe data for numerical features
df.describe()

"""# Exploratory Data Analysis (EDA)"""

#Age Of the client

sns.histplot(df["age"]  )
plt.show()

sns.boxplot(df["age"] , orient = "h")
plt.show()

new_age = df["age"][df["age"]<= 70]
sns.histplot(new_age )
plt.show()

""" job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","""

#job

sns.countplot(df, x="job")
plt.xticks(rotation= 90 )
plt.show()

#Marital

sns.countplot(df , x = "marital")
plt.xticks(rotation = 90)
plt.show()

#Education

sns.countplot(df , x = "education")
plt.xticks(rotation = 90)
plt.show()

"""default: has credit in default? (binary: "yes","no")"""

#Default

sns.countplot(df , x = "default")
plt.xticks(rotation = 90)
plt.show()

#Balance

sns.histplot(df["balance"] , bins = 100)
plt.show()

sns.boxplot(df["balance"] , orient = "h")
plt.show()

balance_data = df["balance"]

Q1 = np.percentile(balance_data, 25)
Q3 = np.percentile(balance_data, 75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

print(Q1)
print(Q3)

print("Lower whisker:", lower_whisker)
print("Upper whisker:", upper_whisker)

#Housing
sns.countplot(df , x = "housing")
plt.xticks(rotation = 90)
plt.show()

housing_counts = df['housing'].value_counts()
plt.figure(figsize = (6,6))
plt.pie(housing_counts, labels=housing_counts.index, autopct='%1.1f%%', startangle=45)
plt.show()

# Loan -> personal

sns.countplot(df , x = "loan")
plt.xticks(rotation = 90)
plt.show()

#Contact  communication type (categorical: "unknown","telephone","cellular")

sns.countplot(df , x = "contact")
plt.xticks(rotation = 90)
plt.show()

#Day : Last contact day of the month

sns.histplot(df["day"] , bins = 31)
plt.show()

#Month: Last contact month of year

sns.countplot(df , x = "month")
plt.xticks(rotation = 90)
plt.show()

#Duration: Last contact duration, in seconds

sns.countplot(df , x = "month")
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(df["duration"] , orient = "h")
plt.show()

#Calculating the First and Third Quartiles (Q1 and Q3):
duration_data = df["duration"]

#Q1: This is the 25th percentile of the duration_data. It represents the value below which 25% of the data falls.
Q1 = np.percentile(duration_data, 25)

#Q3: This is the 75th percentile of the duration_data. It represents the value below which 75% of the data falls.
Q3 = np.percentile(duration_data, 75)

#Calculating the Interquartile Range (IQR):
#The Interquartile Range (IQR) is the difference between the third quartile (Q3) and the first quartile (Q1).
#It measures the spread of the middle 50% of the data

IQR = Q3 - Q1
upper_whisker = Q3 + 1.5 * IQR

print("Upper whisker:", upper_whisker)

sns.histplot(df["duration"][df["duration"] <= 643])
plt.show()

"""number of contacts performed during this campaign and for this client"""

#Campaign :
sns.histplot(df["campaign"] , bins = 63)
plt.show()

sns.boxplot(df["campaign"] , orient = "h")
plt.show()

duration_data = df["campaign"]

Q1 = np.percentile(duration_data, 25)
Q3 = np.percentile(duration_data, 75)

IQR = Q3 - Q1
upper_whisker = Q3 + 1.5 * IQR

print("Upper whisker:", upper_whisker)

sns.histplot(df["campaign"][df["campaign"] <= 6] , bins=6)
plt.show()

#pdays

sns.histplot(df["pdays"])
plt.show()

positive_pdays_count = df["pdays"][df["pdays"] > -1].count()
negative_pdays_count = df["pdays"][df["pdays"] == -1].count()

labels = ['previously contacted', "not previously contacted"]
counts = [positive_pdays_count, negative_pdays_count]

plt.figure(figsize=(8, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.show()

sns.boxplot(df["pdays"] , orient = "h")
plt.show()

pdays_data = df["pdays"]

Q1 = np.percentile(pdays_data, 25)
Q3 = np.percentile(pdays_data, 75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower whisker:", lower_whisker)
print("Upper whisker:", upper_whisker)

sns.boxplot(x = df["pdays"][df["pdays"] > -1] , orient="h") #41
plt.show()

new_pdays_data = df["pdays"][df["pdays"] > -1]

Q1 = np.percentile(new_pdays_data, 25)
Q3 = np.percentile(new_pdays_data, 75)

IQR = Q3 - Q1

upper_whisker = Q3 + 1.5 * IQR

print("Upper whisker:", upper_whisker)

sns.histplot(df["pdays"][(df["pdays"] > -1) &(df["pdays"] <= 618 )] , bins = 619)
plt.show()

"""number of contacts performed before this campaign and for this client"""

#previous

sns.histplot(df["previous"])
plt.show()

positive_previous_count = df["previous"][df["previous"] > 0].count()
zero_previous_count = df["previous"][df["previous"] == 0].count()

labels = ['previously contacted', "not previously contacted"]
counts = [positive_previous_count, zero_previous_count]

plt.figure(figsize=(8, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.show()

sns.boxplot(x = df["previous"][df["previous"] > 0] , orient="h")
plt.show()

new_previous_data = df["previous"][df["previous"] > 0]

Q1 = np.percentile(new_previous_data, 25)
Q3 = np.percentile(new_previous_data, 75)

IQR = Q3 - Q1

upper_whisker = Q3 + 1.5 * IQR

print("Upper whisker:", upper_whisker)

sns.histplot(df["previous"][(df["previous"] > 0) &(df["previous"] <= 8 )] , bins = 8)
plt.show()

#poutcome
sns.countplot(df , x = "poutcome")
plt.xticks(rotation = 90)
plt.show()

df["poutcome"].value_counts(normalize = True)*100

poutcome_counts = df['poutcome'].value_counts()
plt.figure(figsize = (6,6))
plt.pie(poutcome_counts, labels=poutcome_counts.index, autopct='%1.1f%%')
plt.show()

#y

sns.countplot(df , x = "y")
plt.xticks(rotation = 90)
plt.show()

df["y"].value_counts()

df_copy = df.copy()                                          # a copy of the original DataFrame df and stores it in df_copy. This ensures that any modifications made to df_copy do not affect the original DataFrame
df_copy['y'] = df_copy['y'].replace({"yes": 1 , "no": 0})    # This line replaces 'yes' with 1 and 'no' with 0 to convert the target variable into a numerical format. This is necessary for calculating correlations, which require numerical data.
numeric_data = df_copy.select_dtypes(include=[np.number])    # selects only the numerical columns from df_copy and stores them in numeric_data. The select_dtypes method filters columns based on their data types. In this case, it includes only numerical columns.

#Calculating the Correlation Matrix:
correlation_matrix = numeric_data.corr()
correlation_matrix

"""The correlation matrix shows the pairwise correlation coefficients between the features. The values range from -1 to 1, where:
1 indicates a perfect positive correlation,
-1 indicates a perfect negative correlation,
0 indicates no correlation.**
"""

sns.heatmap(correlation_matrix, annot =True)
plt.show()

"""# Preprocessing"""

#imports

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df['default'] = df['default'].replace({"yes": 1 , "no": 0})
df['housing'] = df['housing'].replace({"yes": 1 , "no": 0})
df['loan'] = df['loan'].replace({"yes": 1 , "no": 0})
df['y'] = df['y'].replace({"yes": 1 , "no": 0})

"""Objective: This line of code converts the previous column into binary values (0 and 1).
apply method: This method applies a function along the axis of the DataFrame. In this case, it applies a lambda function to each value in the previous column.
Lambda function: A small anonymous function defined using the lambda keyword. The function checks the value of x:
If x is 0, it returns 0.
If x is not 0, it returns 1.
Result: The previous column originally may contain multiple values indicating the number of times a customer was previously contacted. After this transformation, it will only have two values:
0: If the customer was never contacted before.
1: If the customer was contacted at least once before.
"""

#convert previous to 1/0 and dropping pdays
df['previous'] = df['previous'].apply(lambda x: 0 if x == 0 else 1)
df.drop(columns=['pdays'], inplace=True)

#label encoding of the ordinal categorical features (education and month)
df['education'] = df['education'].replace({"tertiary": 2 , "secondary": 1 , "primary" :0})

df['month'] = df['month'].replace({"jan": 1, "feb": 2, "mar" :3,
                                   "apr":4 ,"may": 5, "jun":6,
                                   "jul":7 , "aug":8, "sep": 9,
                                   "oct": 10, "nov": 11 , "dec":12})

#one hot encoding of the remaining categorical features
df = pd.get_dummies(df, columns=["job", "marital", "poutcome" , "contact"]  , drop_first = True , dtype = int )

df.head()

"""#outlier handling"""

# Convert 'unknown' values to NaN before scaling
df = df.replace('unknown', np.nan)  # This line of code replaces all occurrences of the string 'unknown' in the DataFrame df with NaN (Not a Number).

#use minmax scaler
#MinMaxScaler: This scaler from sklearn.preprocessing scales each feature to a given range, usually between 0 and 1. This is useful for normalizing the features so that they have similar scales.
columns = df.columns
scaler = MinMaxScaler()

# Handle missing values (NaN) before scaling
# Here, we'll use a simple imputation strategy (filling with the mean)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # strategy='mean': Specifies that missing values should be replaced with the mean of the column.
df_imputed = imputer.fit_transform(df) #Fit and Transform: The fit_transform method is applied to the DataFrame df, which computes the mean of each column and replaces the NaN values with the corresponding column mean.

# Now apply the scaler
df_scaled = scaler.fit_transform(df_imputed)

# Convert back to DataFrame
df = pd.DataFrame(df_scaled , columns=columns)

df.head()

"""#Data augmentation since the dataset is not balanced

it deals with splitting the dataset into training and testing sets.

y: The target variable is extracted from the DataFrame df and stored in y. Here, the target variable is the column "y", which we assume contains the labels we want to predict.

X: The features are extracted by dropping the target column "y" from the DataFrame df. The remaining columns are stored in X.

train_test_split: This function from sklearn.model_selection splits the dataset into training and testing sets.
X: The features.
y: The target variable.
test_size=0.2: Specifies that 20% of the data should be used for testing, and the remaining 80% for training.
random_state=42: This parameter ensures reproducibility. Setting a seed for the random number generator ensures that the results are the same every time the code is run.
X_train: The training set features.
X_test: The testing set features.
y_train: The training set target variable.
y_test: The testing set target variable.
"""

#use SMOTE(Synthetic Minority Over-sampling Technique) on only train data since test results should be based on real data.

#Comment on SMOTE: Mentions that SMOTE will be used to balance the training data, but it will be applied only to the training set to keep the test set as a real-world representative sample.

y = df["y"]
X = df.drop("y" , axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42) #X- innput , y- output

X_train.head(5)

X_test.head(5)

y_train.head(5)

y_test.head(5)

sm = SMOTE()
X_train , y_train = sm.fit_resample(X_train, y_train)
y_train.value_counts()

y_test.value_counts()

X_train.shape

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def plot_confusion_matrix(y_test , y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=cm, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

"""#logistic regression"""

# Logistic Regression hyperparameter tuning
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid_search_lr = GridSearchCV(LogisticRegression(max_iter=200, random_state=42), param_grid_lr, cv=5, scoring='roc_auc')
grid_search_lr.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best cross-validation score for Logistic Regression:", grid_search_lr.best_score_)

# Evaluate the best model on the test set
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_test)
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Test ROC AUC:", roc_auc_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

lr_model = LogisticRegression(max_iter=200 , random_state = 42)
lr_model.fit(X_train,y_train)
y_predlr = lr_model.predict(X_test)

plot_confusion_matrix(y_test , y_predlr)
print(classification_report(y_test, y_predlr))

"""6673 - True Negatives
863 - True Positives
228 - False Negative
1279 - False positive

#Random Forest
"""

# Random Forest hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='roc_auc')
grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best cross-validation score for Random Forest:", grid_search_rf.best_score_)

# Evaluate the best model on the test set
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Test ROC AUC:", roc_auc_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

rf_model = RandomForestClassifier(n_estimators = 100 , random_state = 42)
rf_model.fit(X_train, y_train)
y_predrf = rf_model.predict(X_test)

plot_confusion_matrix(y_test , y_predrf)
print(classification_report(y_test, y_predrf))

"""#XGBoost"""

# XGBoost hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search_xgb = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='logloss'), param_grid_xgb, cv=5, scoring='roc_auc')
grid_search_xgb.fit(X_train, y_train)

print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best cross-validation score for XGBoost:", grid_search_xgb.best_score_)

# Evaluate the best model on the test set
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
print("XGBoost Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Test ROC AUC:", roc_auc_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

xgb_model = xgb.XGBClassifier(eta = 0.25 , n_estimators = 100 , max_depth = 6, random_state = 42)
xgb_model.fit(X_train, y_train)
y_predxgb = xgb_model.predict(X_test)

plot_confusion_matrix(y_test , y_predxgb)
print(classification_report(y_test, y_predxgb))

