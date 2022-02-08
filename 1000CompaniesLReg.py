#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading datasets and extracting dependent and independent variables
dataset=pd.read_csv("1000_Companies.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
print(dataset.head())
#graphing dataset
sns.heatmap(dataset.corr())
#plt.show()

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
#oneHotEncoder = OneHotEncoder(categorical_features=[3])
oneHotEncoder = OneHotEncoder()
X = oneHotEncoder.fit_transform(X).toarray()

#avoiding dummy variable traps
X = X[:, 1:]

#splitting the data into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(linearRegression.fit(X_train,y_train))
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

#testing the model
y_predict = linearRegression.predict(X_test)

#finding coefficients and intercept
print(linearRegression.coef_)
print(linearRegression.intercept_)

#Evaluating model// calculating the R squared value
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))




