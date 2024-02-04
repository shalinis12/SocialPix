#!/usr/bin/env python
# coding: utf-8

# # 1. Install Dependencies and Setup

# In[1]:


get_ipython().system('pip install tensorflow opencv-python matplotlib')


# In[2]:


get_ipython().system('pip list')


# In[3]:


import tensorflow as tf
import os


# In[4]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[5]:


tf.config.list_physical_devices('GPU')


# # 2. Remove dodgy images

# In[10]:


import cv2
import imghdr


# In[11]:


data_dir = 'data' 


# In[12]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[13]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
#             os.remove(image_path)


# # 3. Load Data

# In[14]:


import numpy as np
from matplotlib import pyplot as plt


# In[15]:


data = tf.keras.utils.image_dataset_from_directory('data')


# In[16]:


data_iterator = data.as_numpy_iterator()


# In[17]:


batch = data_iterator.next()


# In[18]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# # 4. Scale Data

# In[19]:


data = data.map(lambda x,y: (x/255, y))


# In[20]:


data.as_numpy_iterator().next()


# # 5. Split Data

# In[21]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[22]:


train_size


# In[23]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# # 6. Build Deep Learning Model

# In[24]:


train


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[26]:


model = Sequential()


# In[27]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[28]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[29]:


model.summary()


# # 7. Train

# In[30]:


logdir='logs'


# In[31]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[32]:


hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])


# # 8. Plot Performance

# In[33]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[34]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# # 9. Evaluate

# In[35]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[36]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[37]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[38]:


print(pre.result(), re.result(), acc.result())


# # 10. Test

# In[39]:


import cv2


# In[57]:


img = cv2.imread('IMG_20151128_115506.jpg')
plt.imshow(img)
plt.show()


# In[58]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[59]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[60]:


yhat


# In[61]:


if yhat > 0.5: 
    print(f'Predicted class is place')
else:
    print(f'Predicted class is food')


# # 11. Save the Model

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save(os.path.join('models','imageclassifier.h5'))


# In[ ]:


new_model = load_model('imageclassifier.h5')


# In[ ]:


new_model.predict(np.expand_dims(resize/255, 0))


# In[ ]:




