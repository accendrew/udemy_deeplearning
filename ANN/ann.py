# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode Male/Female to number
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Convert three country values (Spain, France, Germany) into boolean vales.
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder="passthrough")
X = ct.fit_transform(X)
# Remove dummy variable, only need 2 to uniquely determine country
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import Keras package and libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# First hidden layer, with 11 inputs
classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu'))

# Second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predict test set results
y_pred = classifier.predict(X_test)

# Normalize probabilities to True/False, at 50% threshold
y_pred = (y_pred > 0.5)

# Compare predictions vs actual data
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Accuracy of predictions
y_accuracy = (cm[0, 0] + cm[1, 1]) / len(y_test)
