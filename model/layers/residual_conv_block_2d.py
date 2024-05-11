import tensorflow as tf
from keras import layers, activations

class ResidualConvBlock2D(layers.Layer):
  def __init__(
      self, 
      filters: int, 
      kernel_size: int,
      num_conv_layers: int, 
      padding: str, 
      activation: str, 
      strides: tuple[int, int]=(1, 1), 
      *args, 
      **kwargs
    ):
    super(ResidualConvBlock2D, self).__init__(*args, **kwargs)
    self._name = kwargs.get('name')
    self.conv_block = [
      layers.Conv2D(
        filters=filters, 
        kernel_size=kernel_size, 
        strides=strides, 
        padding=padding,
        activation=None,
        name=f'{self._name}_conv_{i}' 
      ) for i in range(num_conv_layers)
    ]

    self.batch_norm = layers.BatchNormalization()

    self.residual_conv = layers.Conv2D(
      filters=filters, 
      kernel_size=(1, 1), 
      strides=strides, 
      padding=padding,
      activation=None,
      name=f'{self._name}_residual_conv'
    )

    if activation == 'relu':
      self.activation = activations.relu
    elif activation == 'tanh':
      self.activation = activations.tanh
    elif activation == 'sigmoid':
      self.activation = activations.sigmoid
    elif activation == 'leaky_relu':
      self.activation = layers.LeakyReLU()
    else:
      raise ValueError(f'Unknown activation function {activation}.')
    
    self.max_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    self.add = layers.Add()
  
  def call(self, inputs):
    for l in self.conv_block:
      x = l(inputs)

    x = self.batch_norm(x)
    x = self.activation(x)

    residuals = self.residual_conv(inputs)
    x = self.add([x, residuals])
    x = self.max_pool(x)
    return x