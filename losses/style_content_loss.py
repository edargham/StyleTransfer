import tensorflow as tf
from keras import losses

class StyleContentLoss(losses.Loss):
  def __init__(
    self,
    style_weight: float,
    content_weight: float,
    num_style_layers: int,
    num_content_layers: int,
    *args, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.num_style_layers = num_style_layers
    self.num_content_layers = num_content_layers

  def call(self, outputs, targets):
    content_outputs = outputs['content']
    content_targets = targets['content']

    style_outputs = outputs['style']
    style_targets = targets['style']

    style_loss = tf.add_n([
      tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
      for name in style_outputs.keys()
    ])
    style_loss *= self.style_weight / self.num_style_layers

    content_loss = tf.add_n([
      tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
      for name in content_outputs.keys()
    ])
    content_loss *= self.content_weight / self.num_content_layers

    loss = style_loss + content_loss

    return loss



