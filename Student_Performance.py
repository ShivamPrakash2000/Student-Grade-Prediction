#!/usr/bin/env python
# coding: utf-8
"""
Student's grade Prediction
Predict Student's grade based on **studytime**, **age**, **traveltime**, **absences**, **G1**, **G2**
<br/><br/>
**studytime**:- Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
<br/>
**age**:- Age (numeric: from 15 to 22)
<br/>
**traveltime**:- home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
<br/>
**absences**:- number of school absences (numeric: from 0 to 93)
<br/>
 **G1**:- first period grade (numeric: from 0 to 20)
<br/>
**G2**:- Second period grade (numeric: from 0 to 20)
"""

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read Dataset using Pandas
df = pd.read_csv('student-mat.csv', sep=';')

df.head()

# Extract datas from dataset
dependecy = df[["studytime", "age", "traveltime", "absences", "G1", "G2"]]
final_grade = df.G3

#Spliting the dataset into Training set and Test Set
x_train, x_test, y_train, y_test = train_test_split(dependecy, final_grade, test_size= 0.2, random_state=0)

# reference of LinearRegression
model = linear_model.LinearRegression()

# Train the model
model.fit(x_train,y_train)

# Predicted values of test case
y_predicted = model.predict(x_test)

print("Mean squared error is: " + str(mean_squared_error(y_test, y_predicted)))
print("Weights: " + str(model.coef_))
print("intersect: "+ str(model.intercept_))

stdtime = float(input("Enter Study Time:- "))

age = float(input("Enter student's age:- "))

trvtime = float(input("Enter travel time:- "))

absences = float(input("Enter school absences:- "))

g1_score = float(input("Enter first period grade:- "))

g2_score = float(input("Enter second period grade:- "))

op = model.predict(np.array([[stdtime, age, trvtime, absences, g1_score, g2_score]]))

print("Predicted Grade is: " + str(op[0]))
