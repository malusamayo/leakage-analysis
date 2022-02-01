#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.dpi"] = 200
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler


# In[11]:


import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create graph: model
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# create graph: loss
loss = tf.reduce_mean(tf.square(y - y_data))

# bind optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# run graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# In[16]:


from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


# In[ ]:





# In[17]:


model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))


# In[128]:


model = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax'),
])


# In[189]:


model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


# In[190]:


model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])


# In[191]:


model.summary()


# In[192]:


from keras.datasets import mnist
import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[193]:


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[194]:


model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)


# In[197]:


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))


# In[134]:


# recreating the model seems the only way to reset?
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])


# In[135]:


model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[136]:


model = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_callback = model.fit(X_train, y_train, batch_size=128,
                             epochs=100, verbose=1, validation_split=.1)


# In[164]:


def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")


# In[144]:


df = pd.DataFrame(history_callback.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")


# In[28]:


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

def make_model(optimizer="adam", hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

param_grid = {'epochs': [1, 5, 10],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)


# In[29]:


grid.fit(X_train, y_train)


# In[33]:


res = pd.DataFrame(grid.cv_results_)
res.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])


# In[165]:



model = Sequential([
    Dense(1024, input_shape=(784,), activation='relu'),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128,
                    epochs=20, verbose=1, validation_split=.1)


# In[186]:


score = model.evaluate(X_test, y_test, verbose=0)


# In[187]:


score


# In[179]:


model.summary()


# In[178]:


df = pd.DataFrame(history.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")


# In[163]:


from keras.layers import Dropout

model_dropout = Sequential([
    Dense(1024, input_shape=(784,), activation='relu'),
    Dropout(.5),
    Dense(1024, activation='relu'),
    Dropout(.5),
    Dense(10, activation='softmax'),
])
model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_dropout = model_dropout.fit(X_train, y_train, batch_size=128,
                            epochs=20, verbose=1, validation_split=.1)


# In[152]:


df = pd.DataFrame(history_dropout.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")


# In[166]:


score = model.evaluate(X_test, y_test, verbose=0)


# In[167]:


score


# # Batch Normalization

# In[198]:


from keras.layers import BatchNormalization

model_bn = Sequential([
    Dense(512, input_shape=(784,)),
    BatchNormalization(),
    Activation("relu"),
    Dense(512),
    BatchNormalization(),
    Activation("relu"),
    Dense(10, activation='softmax'),
])
model_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_bn = model.fit(X_train, y_train, batch_size=128,
                    epochs=10, verbose=1, validation_split=.1)


# In[199]:


plot_history(history)


# # Convolutions

# In[61]:


from scipy.ndimage import convolve
rng = np.random.RandomState(2)
signal = np.cumsum(rng.normal(size=200))
plt.plot(signal)


# In[100]:


gaussian_filter = np.exp(-np.linspace(-2, 2, 15) ** 2)
gaussian_filter /= gaussian_filter.sum()
plt.plot(gaussian_filter)
gaussian_filter


# In[101]:


plt.plot(signal)
plt.plot(convolve(signal, gaussian_filter))


# In[110]:


from scipy.misc import imread
image = imread("IMG_20170207_090931.jpg")
plt.imshow(image)


# In[111]:


gaussian_2d = gaussian_filter * gaussian_filter[:, np.newaxis]
plt.matshow(gaussian_2d)


# In[107]:


out = convolve(image, gaussian_2d[:, :, np.newaxis])


# In[112]:


plt.imshow(out)


# In[118]:


gray_image = image.mean(axis=2)
plt.imshow(gray_image, cmap="gray")


# In[125]:


gradient_2d = convolve(gaussian_2d, [[-1, 1]])


# In[126]:


plt.imshow(gradient_2d)


# In[127]:


edges = convolve(gray_image, gradient_2d)
plt.imshow(edges, cmap="gray")


# # CNN

# In[174]:


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


X_train_images = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
X_test_images = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[175]:


from keras.layers import Conv2D, MaxPooling2D, Flatten

num_classes = 10
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))


# In[177]:


cnn.summary()


# In[176]:


cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(X_train_images, y_train,
                      batch_size=128, epochs=20, verbose=1, validation_split=.1)


# In[188]:


plot_history(history_cnn)


# In[200]:


cnn.evaluate(X_test_images, y_test)


# In[202]:


df = pd.DataFrame(history_cnn.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
plt.ylim(.9, 1)


# In[205]:


layer1 = cnn.layers[0]


# In[214]:


weights, biases = layer1.get_weights()


# In[215]:


weights.shape


# In[217]:


fig, axes = plt.subplots(4, 6)
for ax, weight in zip(axes.ravel(), weights.T):
    ax.imshow(weight[0, :, :])


# In[218]:


from keras.layers import Conv2D, MaxPooling2D, Flatten

num_classes = 10
cnn = Sequential()
cnn.add(Conv2D(8, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(8, (5, 5), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))


# In[232]:


cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(X_train_images, y_train,
                      batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[233]:


weights, biases = cnn.layers[0].get_weights()
fig, axes = plt.subplots(2, 4)
mi, ma = weights.min(), weights.max()
for ax, weight in zip(axes.ravel(), weights.T):
    ax.imshow(weight[0, :, :].T, vmin=mi, vmax=ma)


# In[223]:


weights.shape


# In[226]:


plt.imshow(weights[:, :, 0, 0])


# In[236]:


asdf = cnn.get_input_at(0)


# In[262]:


from keras import backend as K

# with a Sequential model
get_1rd_layer_output = K.function([cnn.layers[0].input],
                                  [cnn.layers[0].output])
get_3rd_layer_output = K.function([cnn.layers[0].input],
                                  [cnn.layers[3].output])

layer1_output = get_1rd_layer_output([X_train_images[:5]])[0]
layer3_output = get_3rd_layer_output([X_train_images[:5]])[0]


# In[263]:


layer1_output.shape


# In[264]:


layer3_output.shape


# In[267]:


weights, biases = cnn.layers[0].get_weights()
n_images = layer1_output.shape[0]
n_filters = layer1_output.shape[3]
fig, axes = plt.subplots(n_images * 2, n_filters + 1, subplot_kw={'xticks': (), 'yticks': ()})
for i in range(layer1_output.shape[0]):
    # for reach input image (= 2 rows)
    axes[2 * i, 0].imshow(X_train_images[i, :, :, 0], cmap="gray_r")
    axes[2 * i + 1, 0].set_visible(False)
    for j in range(layer1_output.shape[3]):
        # for each feature map (same number in layer 1 and 3)
        axes[2 * i, j + 1].imshow(layer1_output[i, :, :, j], cmap='gray_r')
        axes[2 * i + 1, j + 1].imshow(layer3_output[i, :, :, j], cmap='gray_r')


# In[268]:


from keras.layers import Conv2D, MaxPooling2D, Flatten

num_classes = 10
cnn_small = Sequential()
cnn_small.add(Conv2D(8, kernel_size=(3, 3),
              activation='relu',
              input_shape=input_shape))
cnn_small.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small.add(Conv2D(8, (3, 3), activation='relu'))
cnn_small.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small.add(Flatten())
cnn_small.add(Dense(64, activation='relu'))
cnn_small.add(Dense(num_classes, activation='softmax'))


# In[286]:


cnn_small.summary()


# In[270]:


cnn_small.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_small = cnn_small.fit(X_train_images, y_train,
                      batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[290]:


weights, biases = cnn_small.layers[0].get_weights()
weights2, biases2 = cnn_small.layers[2].get_weights()
print(weights.shape)
print(weights2.shape)


# In[296]:


fig, axes = plt.subplots(9, 8, figsize=(10, 8), subplot_kw={'xticks': (), 'yticks': ()})
mi, ma = weights.min(), weights.max()
for ax, weight in zip(axes[0], weights.T):
    ax.imshow(weight[0, :, :].T, vmin=mi, vmax=ma)
axes[0, 0].set_ylabel("layer1")
mi, ma = weights2.min(), weights2.max()
for i in range(1, 9):
    axes[i, 0].set_ylabel("layer3")
for ax, weight in zip(axes[1:].ravel(), weights2.reshape(3, 3, -1).T):
    ax.imshow(weight[:, :].T, vmin=mi, vmax=ma)


# In[281]:


from keras import backend as K

get_1rd_layer_output = K.function([cnn_small.layers[0].input],
                                  [cnn_small.layers[0].output])
get_3rd_layer_output = K.function([cnn_small.layers[0].input],
                                  [cnn_small.layers[2].output])

layer1_output = get_1rd_layer_output([X_train_images[:5]])[0]
layer3_output = get_3rd_layer_output([X_train_images[:5]])[0]


# In[282]:


layer1_output.shape


# In[283]:


layer3_output.shape


# In[284]:


weights, biases = cnn.layers[0].get_weights()
n_images = layer1_output.shape[0]
n_filters = layer1_output.shape[3]
fig, axes = plt.subplots(n_images * 2, n_filters + 1, figsize=(10, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i in range(layer1_output.shape[0]):
    # for reach input image (= 2 rows)
    axes[2 * i, 0].imshow(X_train_images[i, :, :, 0], cmap="gray_r")
    axes[2 * i + 1, 0].set_visible(False)
    axes[2 * i, 1].set_ylabel("layer1")
    axes[2 * i + 1, 1].set_ylabel("layer3")
    for j in range(layer1_output.shape[3]):
        # for each feature map (same number in layer 1 and 3)
        axes[2 * i, j + 1].imshow(layer1_output[i, :, :, j], cmap='gray_r')
        axes[2 * i + 1, j + 1].imshow(layer3_output[i, :, :, j], cmap='gray_r')


# # Batch Normalization

# In[346]:


from keras.layers import BatchNormalization

num_classes = 10
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(8, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(8, (3, 3)))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(64, activation='relu'))
cnn_small_bn.add(Dense(num_classes, activation='softmax'))


# In[347]:


cnn_small_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_small_bn = cnn_small_bn.fit(X_train_images, y_train,
                                        batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[362]:


hist_small_bn = pd.DataFrame(history_cnn_small_bn.history)
hist_small = pd.DataFrame(history_cnn_small.history)
hist_small_bn.rename(columns=lambda x: x + " BN", inplace=True)
hist_small_bn[['acc BN', 'val_acc BN']].plot()
hist_small[['acc', 'val_acc']].plot(ax=plt.gca(), linestyle='--', color=[plt.cm.Vega10(0), plt.cm.Vega10(1)])


# In[366]:


from keras.layers import BatchNormalization

num_classes = 10
cnn32 = Sequential()
cnn32.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn32.add(Activation("relu"))
#cnn_small_bn.add(BatchNormalization())
cnn32.add(MaxPooling2D(pool_size=(2, 2)))
cnn32.add(Conv2D(32, (3, 3)))
cnn32.add(Activation("relu"))
#cnn_small_bn.add(BatchNormalization())
cnn32.add(MaxPooling2D(pool_size=(2, 2)))
cnn32.add(Flatten())
cnn32.add(Dense(64, activation='relu'))
cnn32.add(Dense(num_classes, activation='softmax'))


# In[367]:


cnn32.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_32 = cnn32.fit(X_train_images, y_train,
                            batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[368]:


from keras.layers import BatchNormalization

num_classes = 10
cnn32_bn = Sequential()
cnn32_bn.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Conv2D(32, (3, 3)))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Flatten())
cnn32_bn.add(Dense(64, activation='relu'))
cnn32_bn.add(Dense(num_classes, activation='softmax'))


# In[369]:


cnn32_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_32_bn = cnn32_bn.fit(X_train_images, y_train,
                                 batch_size=128, epochs=10, verbose=1, validation_split=.1)


# In[371]:


hist_32_bn = pd.DataFrame(history_cnn_32_bn.history)
hist_32 = pd.DataFrame(history_cnn_32.history)
hist_32_bn.rename(columns=lambda x: x + " BN", inplace=True)
hist_32_bn[['acc BN', 'val_acc BN']].plot()
hist_32[['acc', 'val_acc']].plot(ax=plt.gca(), linestyle='--', color=[plt.cm.Vega10(0), plt.cm.Vega10(1)])
plt.ylim(.8, 1)


# # loading VGG

# In[298]:


from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')


# In[311]:


model.summary()


# In[339]:


vgg_weights, vgg_biases = model.layers[1].get_weights()
vgg_weights.shape


# In[307]:


fig, axes = plt.subplots(8, 8, figsize=(10, 8), subplot_kw={'xticks': (), 'yticks': ()})
mi, ma = vgg_weights.min(), vgg_weights.max()
for ax, weight in zip(axes.ravel(), vgg_weights.T):
    ax.imshow(weight.T)


# In[340]:


plt.imshow(image)


# In[333]:


get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
get_6rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[6].output])

layer3_output = get_3rd_layer_output([[image]])[0]
layer6_output = get_6rd_layer_output([[image]])[0]


# In[334]:


print(layer3_output.shape)
print(layer6_output.shape)


# In[341]:


fig, axes = plt.subplots(2, 8, figsize=(10, 4), subplot_kw={'xticks': (), 'yticks': ()})
for ax, activation in zip(axes.ravel(), layer3_output.T):
    ax.imshow(activation[:, :, 0].T, cmap="gray_r")
plt.suptitle("after first pooling layer")


# In[343]:


fig, axes = plt.subplots(2, 8, figsize=(10, 4), subplot_kw={'xticks': (), 'yticks': ()})
for ax, activation in zip(axes.ravel(), layer6_output.T):
    ax.imshow(activation[:, :, 0].T, cmap="gray_r")
plt.suptitle("after second pooling layer")


# In[384]:


import flickrapi
import json


api_key = u'f770a9e7064fa7f8754b1ed8cc8cda4f'
api_secret = u' 2e750f2d723350c8 '

import flickrapi
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')


# In[455]:


json.loads(flickr.photos.licenses.getInfo().decode("utf-8"))


# In[410]:


def get_url(photo_id="33510015330"):
    response = flickr.photos.getsizes(photo_id=photo_id)
    sizes = json.loads(response.decode('utf-8'))['sizes']['size']
    for size in sizes:
        if size['label'] == "Small":
            return size['source']
            
get_url()


# In[433]:


from IPython.display import HTML
HTML("<img src='https://farm4.staticflickr.com/3803/33510015330_d1fc801d16_m.jpg'>")


# In[ ]:


# you can check the imagenet classes at https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a


# In[469]:


def search_ids(search_string="python", per_page=10):
    photos_response = flickr.photos.search(text=search_string, per_page=per_page, sort='relevance')
    photos = json.loads(photos_response.decode('utf-8'))['photos']['photo']
    ids = [photo['id'] for photo in photos]
    return ids


# In[471]:


ids = search_ids("ball snake", per_page=100)
urls_ball = [get_url(photo_id=i) for i in ids]
img_string = "\n".join(["<img src='{}'>".format(url) for url in urls_ball])
HTML(img_string)


# In[472]:


ids = search_ids("carpet python", per_page=100)
urls_carpet = [get_url(photo_id=i) for i in ids]
img_string = "\n".join(["<img src='{}'>".format(url) for url in urls_carpet])
HTML(img_string)


# In[480]:


get_ipython().system('mkdir -p snakes/carpet')
get_ipython().system('mkdir snakes/ball')


# In[481]:



from urllib.request import urlretrieve
import os
for url in urls_carpet:
    urlretrieve(url, os.path.join("snakes", "carpet", os.path.basename(url)))


# In[482]:


for url in urls_ball:
    urlretrieve(url, os.path.join("snakes", "ball", os.path.basename(url)))


# In[511]:


from keras.applications.vgg16 import preprocess_input


# In[ ]:


from keras.preprocessing import image

images_carpet = [image.load_img(os.path.join("snakes", "carpet", os.path.basename(url)), target_size=(224, 224))
                 for url in urls_carpet]
images_ball = [image.load_img(os.path.join("snakes", "ball", os.path.basename(url)), target_size=(224, 224))
                 for url in urls_ball]
X = np.array([image.img_to_array(img) for img in images_carpet + images_ball])


# In[527]:


from keras.preprocessing import image

images_carpet = [image.load_img(os.path.join("snakes", "carpet", os.path.basename(url)), target_size=(224, 224))
                 for url in urls_carpet]
images_ball = [image.load_img(os.path.join("snakes", "ball", os.path.basename(url)), target_size=(224, 224))
                 for url in urls_ball]
X = np.array([image.img_to_array(img) for img in images_carpet + images_ball])


# In[566]:


fig, axes = plt.subplots(6, 4, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(5, 8))
for img, ax in zip(images_carpet, axes.ravel()):
    ax.imshow(img)


# In[565]:


fig, axes = plt.subplots(6, 4, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(5, 8))
for img, ax in zip(images_ball, axes.ravel()):
    ax.imshow(img)


# In[529]:


X.shape


# In[530]:


from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X)
features = model.predict(X_pre)


# In[531]:


features.shape


# In[532]:


features_ = features.reshape(200, -1)


# In[537]:


from sklearn.model_selection import train_test_split
y = np.zeros(200, dtype='int')
y[100:] = 1
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)


# In[548]:


from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV().fit(X_train, y_train)


# In[549]:


print(lr.score(X_train, y_train))


# In[550]:


print(lr.score(X_test, y_test))


# In[553]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, lr.predict(X_test))


# In[ ]:




