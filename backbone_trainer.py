import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

from model.backbone_autoenc import build_backbone


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

  train_datagen = ImageDataGenerator(
    rescale=1./255.0,  
    validation_split=0.1
  )

  train_set = train_datagen.flow_from_directory(
    directory='../test',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='input',
    shuffle=True,
    batch_size=128,
    subset='training'
  )

  val_set = train_datagen.flow_from_directory(
    directory='../test',
    target_size=(224, 224),
    class_mode='input',
    shuffle=False,
    batch_size=128,
    subset='validation'
  )

  backbone = build_backbone()

  learning_rate_sched = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=len(train_set),
    decay_rate=0.9,
    staircase=True
  )

  loss_fn = keras.losses.MeanSquaredError()
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate_sched)

  backbone.compile(optimizer=optimizer, loss=loss_fn)

  backbone.summary()
  tf.keras.utils.plot_model(backbone, to_file='backbone.png', show_shapes=True, show_layer_names=True)

  backbone.fit(
    train_set,
    epochs=5,
    validation_data=val_set,
    callbacks=callbacks
  )