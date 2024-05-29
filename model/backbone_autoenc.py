import tensorflow as tf
from keras import layers, Model

def build_backbone(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    conv_block_1 = [
      layers.Conv2D(
        filters=64, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block1_conv{i+1}' 
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      #layers.ReLU()
    ]
    
    residual_conv_1 = layers.Conv2D(      
      filters=64, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      kernel_initializer='glorot_uniform',
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
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block2_conv{i+1}' 
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      #layers.ReLU()
    ]
    
    residual_conv_2 = layers.Conv2D(      
      filters=128, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      kernel_initializer='glorot_uniform',
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
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block3_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.ReLU()
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
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block4_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]
    
    residual_conv_4 = layers.Conv2D(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block4_residual_conv',
      kernel_initializer='glorot_uniform',
    )
    adder_4 = layers.Add()
    max_pool_4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    conv_block_5 = [
      layers.Conv2D(
        filters=512, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block5_conv{i+1}' 
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]
    
    residual_conv_5 = layers.Conv2D(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block5_residual_conv',
      kernel_initializer='glorot_uniform',
    )
    adder_5 = layers.Add()
    max_pool_5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    # Start of the decoder part
    up_sampling_1 = layers.UpSampling2D(size=(2, 2))
    conv_block_6 = [
      layers.Conv2DTranspose(
        filters=512, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        kernel_initializer='glorot_uniform',
        name=f'block6_conv{i+1}'
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]

    residual_conv_6 = layers.Conv2DTranspose(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      kernel_initializer='glorot_uniform',
      name=f'block6_residual_conv',
    )
    adder_6 = layers.Add()

    up_sampling_2 = layers.UpSampling2D(size=(2, 2))
    conv_block_7 = [
      layers.Conv2DTranspose(
        filters=512, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        name=f'block7_conv{i+1}',
        kernel_initializer='glorot_uniform',
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]

    residual_conv_7 = layers.Conv2DTranspose(      
      filters=512, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block7_residual_conv',
      kernel_initializer='glorot_uniform',
    )
    adder_7 = layers.Add()

    up_sampling_3 = layers.UpSampling2D(size=(2, 2))
    conv_block_8 = [
      layers.Conv2DTranspose(
        filters=256, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        name=f'block8_conv{i+1}',
        kernel_initializer='glorot_uniform',
      ) for i in range(4)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]

    residual_conv_8 = layers.Conv2DTranspose(      
      filters=256, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block8_residual_conv',
      kernel_initializer='glorot_uniform',
    )
    adder_8 = layers.Add()

    up_sampling_4 = layers.UpSampling2D(size=(2, 2))
    conv_block_9 = [
      layers.Conv2DTranspose(
        filters=128, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        name=f'block9_conv{i+1}',
        kernel_initializer='glorot_uniform',    
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      layers.LeakyReLU()
    ]

    residual_conv_9 = layers.Conv2DTranspose(      
      filters=128, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      name=f'block9_residual_conv'
    )
    adder_9 = layers.Add()

    up_sampling_5 = layers.UpSampling2D(size=(2, 2))
    conv_block_10 = [
      layers.Conv2DTranspose(
        filters=64, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation='relu',
        name=f'block10_conv{i+1}',
        kernel_initializer='glorot_uniform',
      ) for i in range(2)
    ] + [
      layers.BatchNormalization(),
      #layers.LeakyReLU()
    ]

    residual_conv_10 = layers.Conv2DTranspose(      
      filters=64, 
      kernel_size=(1, 1), 
      strides=(1, 1), 
      padding='same',
      activation=None,
      kernel_initializer='glorot_uniform',
      name=f'block10_residual_conv'
    )
    adder_10 = layers.Add()

    conv_out = layers.Conv2DTranspose(
      filters=3,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='same',
      activation=None,
      name='output_conv'
    )

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

    # Start of the decoder part
    x6 = up_sampling_1(x5)
    for l in conv_block_6:
      x6 = l(x6)
    res6 = residual_conv_6(up_sampling_1(x5))
    x6 = adder_6([x6, res6])

    x7 = up_sampling_2(x6)
    for l in conv_block_7:
      x7 = l(x7)
    res7 = residual_conv_7(up_sampling_2(x6))
    x7 = adder_7([x7, res7])

    x8 = up_sampling_3(x7)
    for l in conv_block_8:
      x8 = l(x8)
    res8 = residual_conv_8(up_sampling_3(x7))
    x8 = adder_8([x8, res8])

    x9 = up_sampling_4(x8)
    for l in conv_block_9:
      x9 = l(x9)
    res9 = residual_conv_9(up_sampling_4(x8))
    x9 = adder_9([x9, res9])

    x10 = up_sampling_5(x9)
    for l in conv_block_10:
      x10 = l(x10)
    res10 = residual_conv_10(up_sampling_5(x9))
    x10 = adder_10([x10, res10])

    x = conv_out(x10)

    return Model(inputs=inputs, outputs=[x])
