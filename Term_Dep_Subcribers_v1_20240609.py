# -*- coding: utf-8 -*-
"""Term_Dep_Subcribers_v2_20240609_V3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o6Pd_e-cEea5UgBcWM-HZ8EHqczjIH6w

# Load Required Libraries
"""

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

from google.colab import drive
drive.mount('/content/drive')

"""# Load datasets"""

train_df = pd.read_csv('/content/drive/MyDrive/SEM C/train.csv' , sep = ';')
#test_df = pd.read_csv('/content/drive/MyDrive/SEM C/test.csv' , sep = ';')

#test_df.shape

train_df.head(5)

df = train_df

df.columns

df.nunique()

df.describe()

"""# Exploratory Data Analysis"""

#Age
sns.histplot(df["age"]  )
plt.show()

sns.boxplot(df["age"] , orient = "h")
plt.show()

new_age = df["age"][df["age"]<= 70]
sns.histplot(new_age )
plt.show()

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

#Contact

sns.countplot(df , x = "contact")
plt.xticks(rotation = 90)
plt.show()

#Day
sns.histplot(df["day"] , bins = 31)
plt.show()

#Month

sns.countplot(df , x = "month")
plt.xticks(rotation = 90)
plt.show()

#Duration

sns.countplot(df , x = "month")
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(df["duration"] , orient = "h")
plt.show()

duration_data = df["duration"]

Q1 = np.percentile(duration_data, 25)
Q3 = np.percentile(duration_data, 75)

IQR = Q3 - Q1
upper_whisker = Q3 + 1.5 * IQR

print("Upper whisker:", upper_whisker)

sns.histplot(df["duration"][df["duration"] <= 643])
plt.show()

#Campaign
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

df_copy = df.copy()
df_copy['y'] = df_copy['y'].replace({"yes": 1 , "no": 0})
numeric_data = df_copy.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
correlation_matrix

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
df = df.replace('unknown', np.nan)

#use minmax scaler
columns = df.columns
scaler = MinMaxScaler()

# Handle missing values (NaN) before scaling
# Here, we'll use a simple imputation strategy (filling with the mean)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

# Now apply the scaler
df_scaled = scaler.fit_transform(df_imputed)

# Convert back to DataFrame
df = pd.DataFrame(df_scaled , columns=columns)

df.head()

"""#Data augmentation since the dataset is not balanced"""

#use SMOTE on only train data since test results should be based on real data.
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

lr_model = LogisticRegression(max_iter=200 , random_state = 42)
lr_model.fit(X_train,y_train)
y_predlr = lr_model.predict(X_test)

plot_confusion_matrix(y_test , y_predlr)
print(classification_report(y_test, y_predlr))

"""#Random Forest"""

rf_model = RandomForestClassifier(n_estimators = 100 , random_state = 42)
rf_model.fit(X_train, y_train)
y_predrf = rf_model.predict(X_test)

plot_confusion_matrix(y_test , y_predrf)
print(classification_report(y_test, y_predrf))

"""#XGBoost"""

xgb_model = xgb.XGBClassifier(eta = 0.25 , n_estimators = 100 , max_depth = 6, random_state = 42)
xgb_model.fit(X_train, y_train)
y_predxgb = xgb_model.predict(X_test)

plot_confusion_matrix(y_test , y_predxgb)
print(classification_report(y_test, y_predxgb))

