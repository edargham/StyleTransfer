import keras.metrics
import keras.optimizers
import keras.optimizers.schedules
import keras.optimizers.schedules.learning_rate_schedule
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.preprocessing.image import ImageDataGenerator

from model.backbone_v2 import build_backbone_v2


if __name__ == '__main__':
  model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_backbone.h5',
    monitor='val_loss',
    mode='min',
    save_weights_only=False,
    save_best_only=False
  )

  early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5
  )

  callbacks = [model_checkpoint]

  train, val = tf.keras.datasets.cifar100.load_data()
  x_train, y_train = train
  x_val, y_val = val

  y_train = keras.utils.to_categorical(y_train, 100)
  y_val = keras.utils.to_categorical(y_val, 100)

  train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=20, 
    width_shift_range=0.2,  
    height_shift_range=0.2,
    zoom_range=0.2,  
    shear_range=0.2, 
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True
  )

  val_datagen = ImageDataGenerator(
    rescale=1./255,
  )

  train_set = train_datagen.flow(x_train, y_train, batch_size=32, shuffle=True)
  val_set = val_datagen.flow(x_val, y_val, batch_size=32)

  backbone = build_backbone_v2(num_classes=100)

  learning_rate_sched = keras.optimizers.schedules.learning_rate_schedule.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=len(train_set),
    decay_rate=0.98,
    staircase=True
  )

  loss_fn = keras.losses.CategoricalCrossentropy()
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate_sched)

  backbone.compile(optimizer=optimizer, loss=loss_fn, metrics=[
    keras.metrics.CategoricalAccuracy(), 
    keras.metrics.Precision(), 
    keras.metrics.Recall()
  ])

  backbone.summary()
  tf.keras.utils.plot_model(backbone, to_file='backbone_v2.png', show_shapes=True, show_layer_names=True)

  backbone.fit(
    train_set,
    epochs=500,
    validation_data=val_set,
    callbacks=callbacks
  )