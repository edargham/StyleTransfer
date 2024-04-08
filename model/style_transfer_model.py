import tensorflow as tf
import keras
from keras.callbacks import CallbackList

from tqdm import tqdm
from utils import gram_matrix


class StyleContentModel(keras.Model):
  def __init__(
    self,
    model_backbone: keras.Model, 
    style_layers: list[str], 
    content_layers: list[str],
    total_variation_weight: float,
    *args, 
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.backbone = model_backbone
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.backbone.trainable = False
    self.total_variation_weight = total_variation_weight

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    backbone_out = self.backbone(preprocessed_input)

    style_outputs = backbone_out[:self.num_style_layers]
    content_outputs = backbone_out[self.num_style_layers:]

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
    style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}
  
  @tf.function
  def clip_0_1(self, image):
   return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
  
  @tf.function
  def train_step(self, data):
    input_image, targets = data
    with tf.GradientTape() as tape:
      outputs = self(input_image)
      loss = self.loss(outputs, targets)
      loss += self.total_variation_weight * tf.image.total_variation(input_image)
    gradients = tape.gradient(loss, input_image)
    self.optimizer.apply_gradients([(gradients, input_image)])
    input_image.assign(self.clip_0_1(input_image))
    return loss
  
  def fit(self, x, y, epochs, steps_per_epoch, callbacks=None):
    if callbacks is None:
      callbacks = []

    callbacks = CallbackList(callbacks)
    callbacks.set_model(self)
    callbacks.set_params({
      'epochs': epochs,
      'steps': steps_per_epoch,
      'verbose': 1,
    })

    callbacks.on_train_begin()
    history = []

    print('\nTraining Progress')
    for n in range(epochs):
      callbacks.on_epoch_begin(n)
      
      pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {n+1}/{epochs}', unit='steps')
      for m in pbar:
        loss = self.train_step((x, y))
        metrics = {'loss': loss.numpy()}
        pbar.set_postfix(metrics)
        history.append(metrics)
      
      pbar.close()
      
      callbacks.on_epoch_end(n, logs=metrics) 
    
    callbacks.on_train_end()
    return history

