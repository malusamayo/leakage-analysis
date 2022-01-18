#!/usr/bin/env python
# coding: utf-8

# # Gradient Descent

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ### Exercise 1
# 
# You've just been hired at a wine company and they would like you to help them build a model that predicts the quality of their wine based on several measurements. They give you a dataset with wine
# 
# - Load the ../data/wines.csv into Pandas
# - Use the column called "Class" as target
# - Check how many classes are there in target, and if necessary use dummy columns for a multi-class classification
# - Use all the other columns as features, check their range and distribution (using seaborn pairplot)
# - Rescale all the features using either MinMaxScaler or StandardScaler
# - Build a deep model with at least 1 hidden layer to classify the data
# - Choose the cost function, what will you use? Mean Squared Error? Binary Cross-Entropy? Categorical Cross-Entropy?
# - Choose an optimizer
# - Choose a value for the learning rate, you may want to try with several values
# - Choose a batch size
# - Train your model on all the data using a `validation_split=0.2`. Can you converge to 100% validation accuracy?
# - What's the minumum number of epochs to converge?
# - Repeat the training several times to verify how stable your results are

# In[ ]:


df = pd.read_csv('../data/wines.csv')


# In[ ]:


df.head()


# In[ ]:


y = df['Class']


# In[ ]:


y.value_counts()


# In[ ]:


y_cat = pd.get_dummies(y)


# In[ ]:


y_cat.head()


# In[ ]:


X = df.drop('Class', axis=1)


# In[ ]:


X.shape


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(df, hue='Class')


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()


# In[ ]:


Xsc = sc.fit_transform(X)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K


# In[ ]:


K.clear_session()
model = Sequential()
model.add(Dense(5, input_shape=(13,),
                kernel_initializer='he_normal',
                activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(RMSprop(lr=0.1),
              'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Xsc, y_cat.values,
          batch_size=8,
          epochs=10,
          verbose=1,
          validation_split=0.2)


# ### Exercise 2
# 
# Since this dataset has 13 features we can only visualize pairs of features like we did in the Paired plot. We could however exploit the fact that a neural network is a function to extract 2 high level features to represent our data.
# 
# - Build a deep fully connected network with the following structure:
#     - Layer 1: 8 nodes
#     - Layer 2: 5 nodes
#     - Layer 3: 2 nodes
#     - Output : 3 nodes
# - Choose activation functions, inizializations, optimizer and learning rate so that it converges to 100% accuracy within 20 epochs (not easy)
# - Remember to train the model on the scaled data
# - Define a Feature Function like we did above between the input of the 1st layer and the output of the 3rd layer
# - Calculate the features and plot them on a 2-dimensional scatter plot
# - Can we distinguish the 3 classes well?
# 

# In[ ]:


K.clear_session()
model = Sequential()
model.add(Dense(8, input_shape=(13,),
                kernel_initializer='he_normal', activation='tanh'))
model.add(Dense(5, kernel_initializer='he_normal', activation='tanh'))
model.add(Dense(2, kernel_initializer='he_normal', activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Xsc, y_cat.values,
          batch_size=16,
          epochs=20,
          verbose=1)


# In[ ]:


model.summary()


# In[ ]:


inp = model.layers[0].input
out = model.layers[2].output


# In[ ]:


features_function = K.function([inp], [out])


# In[ ]:


features = features_function([Xsc])[0]


# In[ ]:


features.shape


# In[ ]:


plt.scatter(features[:, 0], features[:, 1], c=y_cat)


# ### Exercise 3
# 
# Keras functional API. So far we've always used the Sequential model API in Keras. However, Keras also offers a Functional API, which is much more powerful. You can find its [documentation here](https://keras.io/getting-started/functional-api-guide/). Let's see how we can leverage it.
# 
# - define an input layer called `inputs`
# - define two hidden layers as before, one with 8 nodes, one with 5 nodes
# - define a `second_to_last` layer with 2 nodes
# - define an output layer with 3 nodes
# - create a model that connect input and output
# - train it and make sure that it converges
# - define a function between inputs and second_to_last layer
# - recalculate the features and plot them

# In[ ]:


from keras.layers import Input
from keras.models import Model


# In[ ]:


K.clear_session()

inputs = Input(shape=(13,))
x = Dense(8, kernel_initializer='he_normal', activation='tanh')(inputs)
x = Dense(5, kernel_initializer='he_normal', activation='tanh')(x)
second_to_last = Dense(2, kernel_initializer='he_normal',
                       activation='tanh')(x)
outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs=inputs, outputs=outputs)

model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Xsc, y_cat.values, batch_size=16, epochs=20, verbose=1)


# In[ ]:


features_function = K.function([inputs], [second_to_last])


# In[ ]:


features = features_function([Xsc])[0]


# In[ ]:


plt.scatter(features[:, 0], features[:, 1], c=y_cat)


# ## Exercise 4 
# 
# Keras offers the possibility to call a function at each epoch. These are Callbacks, and their [documentation is here](https://keras.io/callbacks/). Callbacks allow us to add some neat functionality. In this exercise we'll explore a few of them.
# 
# - Split the data into train and test sets with a test_size = 0.3 and random_state=42
# - Reset and recompile your model
# - train the model on the train data using `validation_data=(X_test, y_test)`
# - Use the `EarlyStopping` callback to stop your training if the `val_loss` doesn't improve
# - Use the `ModelCheckpoint` callback to save the trained model to disk once training is finished
# - Use the `TensorBoard` callback to output your training information to a `/tmp/` subdirectory
# - Watch the next video for an overview of tensorboard

# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# In[ ]:


checkpointer = ModelCheckpoint(filepath="/tmp/udemy/weights.hdf5",
                               verbose=1, save_best_only=True)


# In[ ]:


earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')


# In[ ]:


tensorboard = TensorBoard(log_dir='/tmp/udemy/tensorboard/')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xsc, y_cat.values,
                                                    test_size=0.3,
                                                    random_state=42)


# In[ ]:


K.clear_session()

inputs = Input(shape=(13,))

x = Dense(8, kernel_initializer='he_normal', activation='tanh')(inputs)
x = Dense(5, kernel_initializer='he_normal', activation='tanh')(x)
second_to_last = Dense(2, kernel_initializer='he_normal',
                       activation='tanh')(x)
outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs=inputs, outputs=outputs)

model.compile(RMSprop(lr=0.05), 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32,
          epochs=20, verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, earlystopper, tensorboard])
model.evaluate(X_test, y_test) # added

# Run Tensorboard with the command:
# 
#     tensorboard --logdir <the-folder-you-chose-in-the-callback>
