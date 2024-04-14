import tensorflow as tf
from keras import layers, Model

def build_backbone_v2(num_classes: int, input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)
    conv_block_1 = [
      layers.Conv2D(
        filters=64, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation=None,
        name=f'block1_conv{i+1}' 
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]
    
    residual_conv_1 = layers.Conv2D(      
      filters=64, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block1_residual_conv'
    )
    adder_1 = layers.Add()
    max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_block_2 = [
      layers.Conv2D(
        filters=128, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation=None,
        name=f'block2_conv{i+1}' 
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]
    
    residual_conv_2 = layers.Conv2D(      
      filters=128, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block2_residual_conv'
    )
    adder_2 = layers.Add()
    max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_block_3 = [
      layers.Conv2D(
        filters=256, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation=None,
        name=f'block3_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]
    
    residual_conv_3 = layers.Conv2D(      
      filters=256, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block3_residual_conv'
    )
    adder_3 = layers.Add()
    max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_block_4 = [
      layers.Conv2D(
        filters=512, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation=None,
        name=f'block4_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]
    
    residual_conv_4 = layers.Conv2D(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block4_residual_conv'
    )
    adder_4 = layers.Add()
    max_pool_4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_block_5 = [
      layers.Conv2D(
        filters=512, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation=None,
        name=f'block5_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]
    
    residual_conv_5 = layers.Conv2D(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block5_residual_conv'
    )
    adder_5 = layers.Add()
    max_pool_5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    flatten = layers.Flatten()
    
    dense_block = [
      layers.Dense(4096, activation='relu'),
      layers.Dense(4096, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
    ]

    x1 = inputs
    for l in conv_block_1:
      x1 = l(x1)
    res1 = residual_conv_1(inputs)
    x1 = adder_1([x1, res1])
    x1 = max_pool_1(x1)

    x2 = x1
    for l in conv_block_2:
      x2 = l(x2)
    res2 = residual_conv_2(x1)
    x2 = adder_2([x2, res2])
    x2 = max_pool_2(x2)

    x3 = x2
    for l in conv_block_3:
      x3 = l(x3)
    res3 = residual_conv_3(x2)
    x3 = adder_3([x3, res3])
    x3 = max_pool_3(x3)

    x4 = x3
    for l in conv_block_4:
      x4 = l(x4)
    res4 = residual_conv_4(x3)
    x4 = adder_4([x4, res4])
    x4 = max_pool_4(x4)

    x5 = x4
    for l in conv_block_5:
      x5 = l(x5)
    res5 = residual_conv_5(x4)
    x5 = adder_5([x5, res5])
    x5 = max_pool_5(x5)

    x = flatten(x5)

    for l in dense_block:
      x = l(x)

    return Model(inputs, x)