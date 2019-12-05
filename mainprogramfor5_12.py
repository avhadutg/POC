#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import re
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


data = pd.read_csv('/home/devops/Downloads/MLOP-s-POC-AWS-master/iris.csv',
names=['sepal_length','sepal_width','petal_length','petal_width','species'])
data.head()


# In[3]:


seaborn.pairplot(data, hue="species", size=2, diag_kind="kde")
plt.show()


# In[4]:


from sklearn.preprocessing import LabelBinarizer
species_lb = LabelBinarizer()
Y = species_lb.fit_transform(data.species.values)


# In[5]:


from sklearn.preprocessing import normalize
FEATURES = data.columns[0:4]
X_data = data[FEATURES].as_matrix()
X_data = normalize(X_data)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)
X_train.shape


# In[7]:


import tensorflow as tf
# Parameters
learning_rate = 0.01
training_epochs = 15


# In[8]:


# Neural Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 1st layer number of neurons
n_input = X_train.shape[1] # input shape (105, 4)
n_classes = y_train.shape[1] # classes to predict


# In[9]:


# Inputs
X = tf.placeholder("float", shape=[None, n_input])
y = tf.placeholder("float", shape=[None, n_classes])
# Dictionary of Weights and Biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[10]:


# Model Forward Propagation step
def forward_propagation(x):
    # Hidden layer1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] 
    return out_layer
# Model Outputs
yhat = forward_propagation(X)
ypredict = tf.argmax(yhat, axis=1)


# In[11]:


# Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)


# In[12]:


# Initializing the variables
init = tf.global_variables_initializer()
from datetime import datetime
startTime = datetime.now()
with tf.Session() as sess:
    sess.run(init)
    
    #writer.add_graph(sess.graph)
    #EPOCHS
    for epoch in range(training_epochs):
        #Stochasting Gradient Descent
        for i in range(len(X_train)):
            summary = sess.run(train_op, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})
        
        train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train, y: y_train}))
        test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test, y: y_test}))
                
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        #print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))
    sess.close()
print("Time taken:", datetime.now() - startTime)


# In[13]:


# preprocess the data
species={'Setosa':0,'Versicolor':1,'Virginica':2}
data.species=[species[item] for item in data.species]
data


# In[14]:


df = pd.DataFrame(data) 
fun=df[['sepal_length','sepal_width','petal_length','petal_width']]

cls=df[['species']]


# In[15]:


X1_train, X1_test,y1_train, y1_test  = train_test_split(fun,cls, test_size=0.3, random_state=1)

X1_train=X1_train.reset_index(drop=True)
X1_test=X1_test.reset_index(drop=True)
y1_train=y1_train.reset_index(drop=True)
y1_test=y1_test.reset_index(drop=True)
y1_train


# In[16]:


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir="iris_model")


# In[17]:


# Fit model.
classifier.fit(x=X1_train, y=y1_train, steps=2000)


# In[18]:

sepal_len=input("Enter the Sepal length")
sepal_wid=input("Enter the Sepal width")
petal_len=input("Enter the Petal length")
petal_wid=input("Enter the Petal width")

new_samples = np.array(
    [[sepal_len,sepal_wid,petal_len,petal_wid]], dtype=float)


# In[19]:


val=str(list(classifier.predict(new_samples)))


f=open('one.txt','w')
f.write(val)



