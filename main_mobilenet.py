import utils
from config import config
import tensorflow as tf
import keras
import random
import os

from model.style_transfer_model_MobileNet import StyleContentModel
from losses.style_content_loss import StyleContentLoss

if __name__ == '__main__':
    # Create a directory to save the results if it doesn't exist
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    # Run the code for a specified number of times
    num_runs = config['num_runs']
    for i in range(num_runs):
        print(f'Run {i + 1}/{num_runs}')

        mobilenet = utils.load_mobilenet_v2()

        # Load style and content layers randomly from config
        all_mobilenet_layers = config['mobilenet_layers']
        num_layers = len(all_mobilenet_layers)

        # Set the number of style and content layers you want to use randomly
        num_style_layers = random.randint(5,10)
        num_content_layers = random.randint(2,5)

        # Randomly select style layers
        style_layers = random.sample(all_mobilenet_layers, num_style_layers)

        # Randomly select content layers
        content_layers = random.sample(all_mobilenet_layers, num_content_layers)

        style_backbone = utils.get_submodel(mobilenet, style_layers + content_layers)
        style_backbone.summary()

        style_image = utils.load_img(config['style_image_path'])
        content_image = utils.load_img(config['content_image_path'])

        extractor = StyleContentModel(
            model_backbone=style_backbone,
            style_layers=style_layers,
            content_layers=content_layers,
            total_variation_weight=config['total_variation_weight'],
        )

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
        print('Total time: {:.1f}'.format(end - start))

        # Save layer names and output image for each run
        output_dir = os.path.join(config['output_dir'], f'run_{i + 1}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'layer_names.txt'), 'w') as f:
            f.write('Style Layers:\n')
            f.write('\n'.join(style_layers))
            f.write('\n\nContent Layers:\n')
            f.write('\n'.join(content_layers))

        print('Saving image...')
        img_out = utils.tensor_to_image(input_image)
        img_out.save(os.path.join(output_dir, 'output_image.png'), 'PNG')

        print(f'Run {i + 1} completed!\n')