import numpy as np
import pandas as pd
import tensorflow as tf

#Getting the data from csv
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values                #inputs array
y = dataset.iloc[:, -1].values                  #outputs

#encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#split into training set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)
print(xTrain, xTest, yTrain, yTest)


#Building the ANN
ann = tf.keras.models.Sequential()

#adding layers
ann.add(tf.keras.layers.Dense(6, activation='relu'))        #1st hidden layer, 6 neurons
ann.add(tf.keras.layers.Dense(6, activation='relu'))        #2nd hidden layer

ann.add(tf.keras.layers.Dense(1, activation='sigmoid'))     #output layer


#compiling ann
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training ann
ann.fit(xTrain, yTrain, batch_size=32, epochs=50)

### we are given a dataset of a certain customer and asked to find out 
### whether they will stay or go:
#print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))


#predicting the final results
yPredict = ann.predict(xTest)
yPredict = (yPredict > 0.5)
pairs = (np.concatenate((yPredict.reshape(len(yPredict),1), yTest.reshape(len(yTest),1)),1))

#prints names of people likely to leave
testNames = []
data = dataset.iloc[:, 2].values
for i in range(len(pairs)):
    if pairs[i][0]<pairs[i][1]:
        testNames.append(data[i])
#names = [list(dataset)[i][3] for i in range(len(pairs)) if pairs[i][0]<pairs[i][1]]
print(testNames)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(yTest, yPredict)
print(cm)
print(accuracy_score(yTest, yPredict))