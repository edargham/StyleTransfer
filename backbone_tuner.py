import tensorflow as tf
import keras
from keras_tuner import RandomSearch, HyperParameters
from keras.preprocessing.image import ImageDataGenerator

from model.backbone_autoenc import build_backbone


if __name__ == '__main__':
  logdir = './logs'
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

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
    batch_size=96,
    subset='training'
  )

  val_set = train_datagen.flow_from_directory(
    directory='../test',
    target_size=(224, 224),
    class_mode='input',
    shuffle=False,
    batch_size=96,
    subset='validation'
  )

  # backbone = build_backbone(num_classes=100)
  def build_hypermodel(hp: HyperParameters):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    backbone = build_backbone()

    learning_rate_sched = keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=len(train_set),
      decay_rate=0.9,
      staircase=True
    )

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_sched)

    backbone.compile(optimizer=optimizer, loss=loss_fn)

    backbone.summary()
    
    return backbone
  
  tuner = RandomSearch(
    build_hypermodel,
    objective='val_loss',
    max_trials=7,
    executions_per_trial=3,
    directory='tuning_results',
    project_name='model_tuning'
  )

  tuner.search(
    x=train_set,
    validation_data=val_set,
    epochs=1,
    callbacks=[
      tensorboard_callback,
      model_checkpoint
    ]
  )

  best_hps = tuner.get_best_hyperparameters()
  print(
    f"""
      The hyperparameter search is complete.\n
      Optimal hyperparameters:\n
      {best_hps}
    """
  )
  