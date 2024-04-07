import utils
from config import config
import tensorflow as tf
import keras
from PIL.Image import Image
from matplotlib import pyplot as plt
from model.style_transfer_model import StyleContentModel
from losses.style_content_loss import StyleContentLoss

if __name__ == '__main__':
  vgg = utils.load_vgg_19()

  style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
  ]

  content_layers=['block5_conv3']

  style_backbone = utils.get_submodel(vgg, style_layers + content_layers)
  # style_backbone.summary()

  style_image = utils.load_img(config['style_image_path'])
  content_image = utils.load_img(config['content_image_path'])
  # utils.show_content_v_style(content_image, style_image)

  extractor = StyleContentModel(
    model_backbone=style_backbone,
    style_layers=style_layers,
    content_layers=content_layers
  )

  results = extractor(tf.constant(content_image))

  input_image = tf.Variable(content_image)

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

loss_fn = StyleContentLoss(
  style_weight=config['style_weight'],
  content_weight=config['content_weight'],
  num_style_layers=len(style_layers),
  num_content_layers=len(content_layers)
)

opt = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

extractor.compile(optimizer=opt, loss=loss_fn)

_, img = extractor.train_step((input_image, results))
input_image.assign(img)
_, img = extractor.train_step((input_image, results))
input_image.assign(img)
_, img = extractor.train_step((input_image, results))
input_image.assign(img)
img_out = utils.tensor_to_image(input_image)


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    _, img = extractor.train_step((input_image, results))
    input_image.assign(img)
    print(".", end='', flush=True)
  # display.clear_output(wait=True)
  # display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))

img_out.save('./out2.png', 'PNG')

