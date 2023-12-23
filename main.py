import splitfolders
import tensorflow as tf
from tensorflow import keras

# импортируем библиотеку pathlib, а также функцию Path для работы с директориями
import pathlib
from pathlib import Path
import os
# для упорядочивания файлов в директории
import natsort
import scipy

# библиотеки для работы с изображениями
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -- Импорт для подготовки данных: --
# модуль для предварительной обработки изображений
from tensorflow.keras.preprocessing import image
# Класс ImageDataGenerator - для генерации новых изображений на основе имеющихся
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -- Импорт для построения модели: --
# импорт слоев нейросети
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import RMSprop

jack = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart\\jack"
gates = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart\\gates"
modi = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart\\modi"
musk = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart\\musk"
trump = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart\\trump"
# Получим и отсортируем список с названиями фото с женскими лицами


base_path = "G:\\Studing\\ML\\FaceDataSet\\Five_Faces\\standart"

splitfolders.ratio(base_path, 'faces_splited', ratio=(0.8, 0.15, 0.05), seed=18, group_prefix=None )

# определим параметры нормализации данных
train = ImageDataGenerator(rescale=1 / 255)
val = ImageDataGenerator(rescale=1 / 255)

# сгенерируем нормализованные данные
train_data = train.flow_from_directory('faces_splited/train', target_size=(299, 299),
                                       class_mode='categorical', batch_size=3, shuffle=True)
val_data = val.flow_from_directory('faces_splited/val', target_size=(299, 299),
                                   class_mode='categorical', batch_size=3, shuffle=True)



model = Sequential([
    layers.InputLayer(input_shape=(299,299,3)),
    layers.AvgPool2D(2, strides=2),
    layers.Conv2D(32, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(5, activation='softmax')
])
model.summary()

# Файл для сохранения модели с лучшими параметрами
checkpoint_filepath = 'best_model.h5'
# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.000244),
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.000244),
              metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Тренировка модели
history = model.fit(train_data, batch_size=500, verbose=1, epochs=5,
                    validation_data=val_data,
                    callbacks=[model_checkpoint_callback])

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Строим график для accuracy
plt.figure(figsize=(8, 6), edgecolor='blue', facecolor='pink')
plt.plot(range(len(train_accuracy)), train_accuracy, label='Точность тренировочной выборки')
plt.plot(range(len(val_accuracy)), val_accuracy, label='Точность валидационной выборки')
plt.title('Точность')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Строим график для loss
plt.figure(figsize=(8, 6), edgecolor='blue', facecolor='pink')
plt.plot(range(len(train_loss)), train_loss, label='Потери тренировочной выборки')
plt.plot(range(len(val_loss)), val_loss, label='Потери валидационной выборки')
plt.title('Функция потерь')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()