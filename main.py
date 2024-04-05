import utils
from config import config
import tensorflow as tf
import keras
from model.style_transfer_model import StyleContentModel

if __name__ == '__main__':
  vgg = utils.load_vgg_19()

  style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
  ]

  content_layers=['block5_conv2']

  style_backbone = utils.get_submodel(vgg, style_layers + content_layers)
  # style_backbone.summary()

  style_image = utils.load_img(config['style_image_path'])
  content_image = utils.load_img(config['content_image_path'])
  # utils.show_content_v_style(content_image, style_image)

  extractor = StyleContentModel(
    model_backbone=style_backbone,
    style_layers=style_layers,
    content_layers=content_layers,
    content_image=content_image
  )

  results = extractor(tf.constant(content_image))

  print('Styles:')
  for name, output in sorted(results['style'].items()):
    print('  ', name)
    print('    shape: ', output.numpy().shape)
    print('    min: ', output.numpy().min())
    print('    max: ', output.numpy().max())
    print('    mean: ', output.numpy().mean())
    print()

  print('Contents:')
  for name, output in sorted(results['content'].items()):
    print('  ', name)
    print('    shape: ', output.numpy().shape)
    print('    min: ', output.numpy().min())
    print('    max: ', output.numpy().max())
    print('    mean: ', output.numpy().mean())