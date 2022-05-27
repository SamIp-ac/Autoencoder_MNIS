import pandas as pd
import numpy as np
import os
from cv2 import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# Data argumentation
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
# Grid search
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K


cwd = os.getcwd()
os.chdir(os.path.join(cwd, 'Project_AutoEncoder'))
tf.random.set_seed(4012)


def Creat_dataset(word='K', test='Train'):
    os.chdir(os.path.join(cwd, 'Project_AutoEncoder'))
    cwd_K = os.path.join(cwd, 'Project_AutoEncoder', word)
    os.chdir(os.path.join(os.getcwd(), word))
    image_name = []
    image = []

    if test == 'Train':

        for i in range(0, 5):
            filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(i))) if
                        file.endswith('.png')]

            image_name.append(filelist)

        for j in range(len(image_name) - 1):
            cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(j))
            os.chdir(cwd_temp)

            for k in image_name[j]:
                temp = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
                temp = cv2.resize(temp, (28, 28))
                image.append(temp)
            os.chdir(cwd_K)

    elif test == 'Test':

        filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(6))) if
                    file.endswith('.png')]

        image_name.append(filelist)

        cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(6))
        os.chdir(cwd_temp)

        for k in image_name[0]:
            temp = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (28, 28))
            image.append(temp)

    elif test == 'val':

        filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(7))) if
                    file.endswith('.png')]

        image_name.append(filelist)

        cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(7))
        os.chdir(cwd_temp)

        for k in image_name[0]:
            temp = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (28, 28))
            image.append(temp)

    image = np.array(image)
    image = tf.cast(image, tf.float32)
    os.chdir(cwd)

    return image / 255  # Gradient explosion if not divide by 255


def Creat_label(image_data, label):
    size = image_data.shape[0]
    labels = []
    if label == 1:
        labels = np.ones(size)
    elif label == 0:
        labels = np.zeros(size)
    return labels


def DATA_concatenate(data_1, data_2):
    Data = np.array(np.concatenate((data_1, data_2), axis=0))
    return Data


class AE_model:
    def build_model(self, filters=4, img_height=28, img_width=28, channel_size=1):
        tf.random.set_seed(4012)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(img_height, img_width, channel_size)))
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(1, activation='relu'))
        model.add(tf.keras.layers.Dense(img_width*img_height*filters*channel_size, activation='relu'))
        model.add(tf.keras.layers.Reshape((img_width, img_height, filters*channel_size)))
        model.add(tf.keras.layers.Conv2DTranspose(filters, (3, 3), strides=(1, 1), padding='same',
                                                  activation='relu'))
        model.add(
            tf.keras.layers.Conv2DTranspose(channel_size, (3, 3), strides=(1, 1), padding='same', activation='sigmoid'))
        model.add(tf.keras.layers.Reshape((img_width, img_height, channel_size)))

        optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, bs=32, ntry=1, X_val=None, y_val=None):

        datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=False, vertical_flip=False, fill_mode='constant', cval=0.0)
        datagen.fit(X_train)
        model = self.build_model()

        history = model.fit(datagen.flow(X_train, y_train, batch_size=bs), epochs=5,
                            validation_data=(X_val, y_val), verbose=0,
                            steps_per_epoch=X_train.shape[0] // bs)

        self.best_model = model
        self.df_history = pd.DataFrame(history.history)
        self.best_loss = model.evaluate(X_train, y_train)

        for i in range(ntry):
            model = self.build_model()
            history = model.fit(datagen.flow(X_train, y_train, batch_size=bs), epochs=5,
                                validation_data=(X_val, y_val), verbose=0,
                                steps_per_epoch=X_train.shape[0] // bs)

            if model.evaluate(X_train, y_train) < self.best_loss:
                self.best_model = model
                self.df_history = pd.DataFrame(history.history)
                self.best_loss = model.evaluate(X_train, y_train)

        os.chdir(os.path.join(cwd, 'Project_AutoEncoder'))
        # self.best_model.save_weights('AE_weights.h5')
        self.best_model.save('AE_model_K.h5')
        return self.df_history, self.best_loss

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def summary(self):
        return self.best_model.summary()

    def evaluate(self, y_pred, y_test):
        model = self.best_model
        evaluate = model.evaluate(y_pred, y_test)
        return evaluate


channel = 1
word = 'K'
X_train = Creat_dataset(word=word, test='Train')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], channel))

X_val = Creat_dataset(word=word, test='val')
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], channel))

X_test = Creat_dataset(word=word, test='Test')
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], channel))

noise_factor = 0.2
X_train_noisy = X_train + noise_factor * tf.random.normal(shape=X_train.shape)
X_val_noisy = X_val + noise_factor * tf.random.normal(shape=X_val.shape)
X_test_noisy = X_test + noise_factor * tf.random.normal(shape=X_test.shape)
# X_train_noisy = tf.clip_by_value(X_train_noisy, clip_value_min=0, clip_value_max=1)

model = AE_model()
'''# define the grid search parameters
model_grid = KerasRegressor(build_fn=model.build_model(), epochs=10, batch_size=32, verbose=0)
parameters = {'neurons': [4, 8, 16, 32, 64]}
grid = GridSearchCV(estimator=model_grid, param_grid=parameters, n_jobs=-1, cv=5)
grid_result = grid.fit(X_val, X_val)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)

model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
# define parameters and values for grid search 
param_grid = {
    'pool_type': ['max', 'average'],
    'conv_activation': ['sigmoid', 'tanh'],    
    'epochs': [n_epochs_cv],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(X_train, to_categorical(y_train))'''

os.chdir(os.path.join(cwd, 'Project_AutoEncoder'))
# plot accuracy on training and validation data
df_history, best_loss = model.train(X_train=X_train_noisy, y_train=X_train, X_val=X_val_noisy, y_val=X_val)

print(df_history)
sns.lineplot(data=df_history[['accuracy', 'val_accuracy']], palette='tab10', linewidth=2.5)
plt.xlabel('no. of epochs')
plt.ylabel('acc')
plt.show()

# If the model is saved, switch it on and not need to fit each time.
# model = tf.keras.models.load_model('AE_model_K.h5')  # Shifted

# del noised
#######################
print(model.summary())
'''print(model.layers)
inp = model.input  # input
outputs = [layer.output for layer in model.layers]
print(outputs)
functors = [K.function([inp], [out]) for out in outputs]
print(list(functors))'''
# prediction
pred = model.predict(X_test_noisy)
image = pred[80]  # 80
image = cv2.resize(image, (224, 224))

# clip
_, clip = cv2.threshold(image*255, 160, 255, type=cv2.THRESH_BINARY)

# original
X_test_backup = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
image_ori = X_test_backup[80, :, :]
image_ori = cv2.resize(image_ori, (224, 224))

# noisy picture
X_test_noisy_backup = np.reshape(X_test_noisy, (X_test_noisy.shape[0], X_test_noisy.shape[1], X_test_noisy.shape[2]))
image_or = X_test_noisy_backup[80, :, :]
image_or = cv2.resize(image_or, (224, 224))
image_all = np.concatenate((image_ori, image_or, image, clip), axis=1)

cv2.imshow('original --> noisy --> output --> Clipped', image_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_2 = pred[90]  # 80
image_2 = cv2.resize(image_2, (224, 224))
image_3 = pred[100]  # 100
image_3 = cv2.resize(image_3, (224, 224))
image_4 = pred[110]  # 110
image_4 = cv2.resize(image_4, (224, 224))
print('The mean of prediction of letter K is ', np.mean(pred))

image_all = np.concatenate((image, image_2, image_3, image_4), axis=1)

cv2.imshow('80 --> 90 --> 100 --> 110', image_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Generate noisy data
temp = noise_factor * tf.random.normal(shape=X_test_noisy.shape)
pred_new = model.predict(temp)
temp = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2]))
temp_1 = temp[0, :, :]
temp_2 = temp[50, :, :]
temp_1, temp_2 = cv2.resize(temp_1, (224, 224)), cv2.resize(temp_2, (224, 224))
pred_new1, pred_new2 = pred_new[0, :, :], pred_new[50, :, :]
pred_new1, pred_new2 = cv2.resize(pred_new1, (224, 224)), cv2.resize(pred_new2, (224, 224))
image_all = np.concatenate((temp_1, pred_new1, temp_2, pred_new2), axis=1)

print('The mean of prediction of noise is ', np.mean(pred_new))
cv2.imshow('noise --> prediction, noise --> prediction', image_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

############
X_test_Y = Creat_dataset(word='Y', test='Test')
X_test_Y = np.reshape(X_test_Y, (X_test_Y.shape[0], X_test_Y.shape[1], X_test_Y.shape[2], channel))
X_test_noisy_Y = X_test_Y + noise_factor * tf.random.normal(shape=X_test_Y.shape)

# predict image Y
pred = model.predict(X_test_noisy_Y)
image = pred[82]
image = cv2.resize(image, (224, 224))

X_test_noisy_Y_backup = np.reshape(X_test_noisy_Y, (X_test_noisy_Y.shape[0], X_test_noisy_Y.shape[1],
                                                    X_test_noisy_Y.shape[2]))
image_or = X_test_noisy_Y_backup[87, :, :]
image_or = cv2.resize(image_or, (224, 224))
image_all = np.concatenate((image_or, image), axis=1)

print('The mean of prediction of Y is ', np.mean(pred))
cv2.imshow('noise --> prediction', image_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

# evaluate
loss, acc = model.evaluate(X_test_noisy, X_test)
print('The loss is {}, acc is {}'.format(loss, acc))

del model
