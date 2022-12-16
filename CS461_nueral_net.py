import keras 
from keras.models import Sequential 
from keras.layers import Dense 
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import csv 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #
from sklearn.preprocessing import OneHotEncoder #
from sklearn.metrics import accuracy_score #
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report   

## prepossess and load data ###########################################
# load data from csv file proccessed_cleavland.csv
data = pd.read_csv('processed_cleveland.csv')

# split data into x and y
x = data.iloc[:,:13].values
y = data.iloc[:,13:14].values

# normalize x data
sc = StandardScaler()
x = sc.fit_transform(x)

# one hot encode y data
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# split data into train:80% and test:10% and validation:10% sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 12)
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

# building the neural network #########################################
model = Sequential()
model.add(Dense(26, input_dim=13, activation= 'relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(7, activation='relu'))
# model.add(Dense(13, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train and test the model #####################################################
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15
, batch_size=30, verbose=1)

# plot the accuracy and loss of the model ########################################
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# validate the model ##################################################
y_pred = model.predict(x_val)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test = list()
for i in range(len(y_val)):
    test.append(np.argmax(y_test[i]))

a = accuracy_score(test, pred)
print("Validation dataset accuracy percentage: ", a*100)

# make a table comparing the predicted and actual outputs
print( "predicted: ", "actual: ")
for i in range(len(pred)):
    print("        ",pred[i], "      ", test[i])





