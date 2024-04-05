import tensorflow as tf
import keras

from utils import gram_matrix


class StyleContentModel(keras.Model):
  def __init__(
    self,
    model_backbone: keras.Model, 
    style_layers: list[str], 
    content_layers: list[str],
    content_image: tf.Tensor,
    *args, 
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.backbone = model_backbone
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.backbone.trainable = False
    self.image = tf.Variable(content_image)

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
  def clip_0_1(image):
   return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
  
  @tf.function
  def train_step(self, data):
    with tf.GradientTape() as tape:
      outputs = self(data)
      loss = self.loss(data, outputs)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return loss

