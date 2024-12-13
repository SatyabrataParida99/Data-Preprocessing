# IMPORT LIBRARY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# iMPORT THE DATASET

dataset = pd.read_csv(r"D:\FSDS Material\Machine Learning Material\11th - ML\5. Data preprocessing\Data.csv")
# Independent & Dependent variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

# sklearn fill missing numerical value 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

# impute categorical value for independent 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0])
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# Impute categorical value for dependent
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 

# Split the data 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=41)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
