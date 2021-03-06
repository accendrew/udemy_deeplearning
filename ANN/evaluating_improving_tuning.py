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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#
# # Evaluate ANN
# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return classifier
#
#
# # K-Fold Cross Validation repeats training and evaluation against revolving subsets of training and test data.
# classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
# mean = accuracies.mean()
# variance = accuracies.std()

# Tuning the ANN
# Apply grid search technique to test the effects of different combinations of model parameters on model accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Optimizer function now parameterized
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# Attempt model training with different values for batch_size, nb_epoch, and optimizer.
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

# Best results, with associated model parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_