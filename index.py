from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv
import sys
import os

columns = ['Id',
           'Elevation',
           'Aspect',
           'Slope',
           'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways',
           'Hillshade_9am',
           'Hillshade_Noon',
           'Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points',
           'Wilderness_Area1',
           'Wilderness_Area2',
           'Wilderness_Area3',
           'Wilderness_Area4',
           'Soil_Type1',
           'Soil_Type2',
           'Soil_Type3',
           'Soil_Type4',
           'Soil_Type5',
           'Soil_Type6',
           'Soil_Type7',
           'Soil_Type8',
           'Soil_Type9',
           'Soil_Type10',
           'Soil_Type11',
           'Soil_Type12',
           'Soil_Type13',
           'Soil_Type14',
           'Soil_Type15',
           'Soil_Type16',
           'Soil_Type17',
           'Soil_Type18',
           'Soil_Type19',
           'Soil_Type20',
           'Soil_Type21',
           'Soil_Type22',
           'Soil_Type23',
           'Soil_Type24',
           'Soil_Type25',
           'Soil_Type26',
           'Soil_Type27',
           'Soil_Type28',
           'Soil_Type29',
           'Soil_Type30',
           'Soil_Type31',
           'Soil_Type32',
           'Soil_Type33',
           'Soil_Type34',
           'Soil_Type35',
           'Soil_Type36',
           'Soil_Type37',
           'Soil_Type38',
           'Soil_Type39',
           'Soil_Type40',
           'Cover_Type']

# Preparing the data for Train and Test.
trainData = pd.read_csv("./data/train.csv", sep=',')
dfTrain = pd.DataFrame(trainData, columns = columns)

testData = pd.read_csv("./data/test.csv", sep=',')
dfTest = pd.DataFrame(testData, columns = columns)

# Encoding for the Train and Test data.
le = preprocessing.LabelEncoder()
for i in columns:
     trainData[i] = le.fit_transform(trainData[i].astype('str'))

y = trainData.Cover_Type
X = trainData[columns]
val_xT = testData[columns]

# Validate and test the model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

val_y_size = val_y.size
# print(val_y_size)
train_y_size = train_y.size
# print(val_y_size)
train_X = [train_X]
train_y = [train_y]
val_X = [val_X]
val_y = [val_y]

# Create and predict with the model
model = LogisticRegression(n_jobs=1, C=1e5)

model.fit(train_X.values.reshape(-1, 1), train_y)

prediction = model.predict(val_xT)

# Validate and test the model
accuTrain = np.sum(model.predict(train_X.values.reshape(-1, 1)) == train_y)/train_y_size
accuTest = np.sum(model.predict(val_X.values.reshape(-1, 1)) == val_y)/val_y_size

# print("Accuracy Train: ", (accuTrain * 100))
# print("Accuracy Test: ", (accuTest * 100))

# Print the output:
output = pd.DataFrame({"PassengerId":  dfTest.Id, "Survived": prediction})
output.to_csv("./data/output.csv", sep=',', index=False)
