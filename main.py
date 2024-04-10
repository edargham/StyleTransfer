import utils
from config import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50

from model.style_transfer_model import StyleContentModel
from losses.style_content_loss import StyleContentLoss

if __name__ == '__main__':
    # Load the ResNet50 model pre-trained on ImageNet data
    resnet = ResNet50(include_top=False, weights='imagenet')

    # Define style and content layers
    style_layers = [
        'conv1_conv',  # Corresponds to block1_conv1 in VGG19
        # Add more layers as per the ResNet architecture
    ]

    content_layers = ['conv5_block3_out']  # Corresponds to block5_conv3 in VGG19

    # Get the submodel from ResNet50
    style_backbone = utils.get_submodel(resnet, style_layers + content_layers)
    style_backbone.summary()

    # Load style and content images
    style_image = utils.load_img(config['style_image_path'])
    content_image = utils.load_img(config['content_image_path'])

    # Create the style content model
    extractor = StyleContentModel(
        model_backbone=style_backbone,
        style_layers=style_layers,
        content_layers=content_layers,
        total_variation_weight=config['total_variation_weight'],
    )

    # Rest of the code remains the same...


  results = extractor(tf.constant(content_image))
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  input_image = tf.Variable(content_image, trainable=True)

  print('Useful information for generating Gram Matrix:')
  print('  Styles:')
  for name, output in sorted(results['style'].items()):
    print('    ', name)
    print('      shape: ', output.numpy().shape)
    print('      min: ', output.numpy().min())
    print('      max: ', output.numpy().max())
    print('      mean: ', output.numpy().mean())
    print()

  print('  Contents:')
  for name, output in sorted(results['content'].items()):
    print('    ', name)
    print('      shape: ', output.numpy().shape)
    print('      min: ', output.numpy().min())
    print('      max: ', output.numpy().max())
    print('      mean: ', output.numpy().mean())

loss_fn = StyleContentLoss(
  style_weight=config['style_weight'],
  content_weight=config['content_weight'],
  num_style_layers=len(style_layers),
  num_content_layers=len(content_layers)
)

opt = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

extractor.compile(optimizer=opt, loss=loss_fn)

import time
start = time.time()

extractor.fit(
  x=input_image,
  y={'style': style_targets, 'content': content_targets},
  epochs=config['epochs'],
  steps_per_epoch=config['steps_per_epoch']
)
  
end = time.time()
print('Total time: {:.1f}'.format(end-start))

print('Saving image...')
img_out = utils.tensor_to_image(input_image)
img_out.save('./out.png', 'PNG')
