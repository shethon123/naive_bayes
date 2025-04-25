#-------------------------------------------------------------------------
# AUTHOR: Sheldin Lau
# FILENAME: naive_bayes.py
# SPECIFICATION: finding the best hyperparameters for a Naive Bayes Classifier
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1 Hour
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
#--> add your Python code here
df = pd.read_csv('weather_training.csv', sep=',', header=0)
data_training = np.array(df.values)[:,-1].astype('f')

X_training = np.array(df.values)[:,1:-1].astype('f')
y_training = np.array(df.values)[:,-1].astype('f')

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
for i in range(len(y_training)):

    difference = 100
    closest = 0
    for j in classes:
        if abs(y_training[i] - j) < difference:
            closest = j
            difference = abs(y_training[i] - j)
    y_training[i] = classes.index(closest)

#reading the test data
#--> add your Python code here
df = pd.read_csv('weather_test.csv', sep=',', header=0)
data_training = np.array(df.values)[:,-1].astype('f')

X_test = np.array(df.values)[:,1:-1].astype('f')
y_test = np.array(df.values)[:,-1].astype('f')

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
for i in range(len(y_test)):
    difference = 100
    closest = 0
    for j in classes:
        if abs(y_test[i] - j) < difference:
            closest = j
            difference = abs(y_test[i] - j)
    y_test[i] = classes.index(closest)

#loop over the hyperparameter value (s)
#--> add your Python code here
highest_accuracy = 0
for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here
    correctPredictions = 0
    for (x_testSample, y_testSample) in zip(X_test, y_test):
        predicted_value = clf.predict([x_testSample])
        real_value = y_testSample
        # print(predicted_value, real_value)
        difference = 100 * abs(predicted_value - real_value) / abs(real_value)
        # print(difference)
        if difference <= 15:
            correctPredictions += 1

    tmp = clf.predict(X_test)
    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here

    accuracy = correctPredictions / len(X_test)
    if accuracy > highest_accuracy:
        print("Highest Naive Bayes accuracy so far: " + str(accuracy) + ", Parameters: s=" + str(s))
        highest_accuracy = accuracy
