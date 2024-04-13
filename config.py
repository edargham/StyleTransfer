config = {
  'style_image_path': './vangogh.jpg',#r'C:\Users\edarg\Pictures\vangogh.jpg',
  'content_image_path': './content.jpeg',#r'C:\Users\edarg\Pictures\Ehmej.JPEG',
  "output_dir": "./MobileNetV2_StyleTransfer",
  'content_weight': [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
  'style_weight': [1e2, 2e2, 3e2, 4e2, 5e2, 1e3, 2e3, 3e3, 4e3, 5e3],
  'total_variation_weight': 0,
  'epochs': 15,
  'steps_per_epoch': 100,
  "num_runs": 25,
  'mobilenet_layers':[
    'block_1_expand',
    'block_1_expand_BN',
    'block_1_expand_relu',
    'block_1_pad',
    'block_1_depthwise',
    'block_1_depthwise_BN',
    'block_1_depthwise_relu',
    'block_1_project',
    'block_1_project_BN',
    'block_2_expand',
    'block_2_expand_BN',
    'block_2_expand_relu',
    'block_2_depthwise',
    'block_2_depthwise_BN',
    'block_2_depthwise_relu',
    'block_2_project',
    'block_2_project_BN',
    'block_2_add',
    'block_3_expand',
    'block_3_expand_BN',
    'block_3_expand_relu',
    'block_3_pad',
    'block_3_depthwise',
    'block_3_depthwise_BN',
    'block_3_depthwise_relu',
    'block_3_project',
    'block_3_project_BN',
    'block_4_expand',
    'block_4_expand_BN',
    'block_4_expand_relu',
    'block_4_depthwise',
    'block_4_depthwise_BN',
    'block_4_depthwise_relu',
    'block_4_project',
    'block_4_project_BN',
    'block_4_add',
    'block_5_expand',
    'block_5_expand_BN',
    'block_5_expand_relu',
    'block_5_depthwise',
    'block_5_depthwise_BN',
    'block_5_depthwise_relu',
    'block_5_project',
    'block_5_project_BN',
    'block_5_add',
    'block_6_expand',
    'block_6_expand_BN',
    'block_6_expand_relu',
    'block_6_pad',
    'block_6_depthwise',
    'block_6_depthwise_BN',
    'block_6_depthwise_relu',
    'block_6_project',
    'block_6_project_BN',
    'block_7_expand',
    'block_7_expand_BN',
    'block_7_expand_relu',
    'block_7_depthwise',
    'block_7_depthwise_BN',
    'block_7_depthwise_relu',
    'block_7_project',
    'block_7_project_BN',
    'block_7_add',
    'block_8_expand',
    'block_8_expand_BN',
    'block_8_expand_relu',
    'block_8_depthwise',
    'block_8_depthwise_BN',
    'block_8_depthwise_relu',
    'block_8_project',
    'block_8_project_BN',
    'block_8_add',
    'block_9_expand',
    'block_9_expand_BN',
    'block_9_expand_relu',
    'block_9_depthwise',
    'block_9_depthwise_BN',
    'block_9_depthwise_relu',
    'block_9_project',
    'block_9_project_BN',
    'block_9_add',
    'block_10_expand',
    'block_10_expand_BN',
    'block_10_expand_relu',
    'block_10_depthwise',
    'block_10_depthwise_BN',
    'block_10_depthwise_relu',
    'block_10_project',
    'block_10_project_BN',
    'block_11_expand',
    'block_11_expand_BN',
    'block_11_expand_relu',
    'block_11_depthwise',
    'block_11_depthwise_BN',
    'block_11_depthwise_relu',
    'block_11_project',
    'block_11_project_BN',
    'block_11_add',
    'block_12_expand_BN',
    'block_12_expand_relu',
    'block_12_depthwise',
    'block_12_depthwise_BN',
    'block_12_depthwise_relu',
    'block_12_project',
    'block_12_project_BN',
    'block_12_add',
    'block_13_expand',
    'block_13_expand_BN',
    'block_13_expand_relu',
    'block_13_pad',
    'block_13_depthwise',
    'block_13_depthwise_BN',
    'block_13_depthwise_relu',
    'block_13_project',
    'block_13_project_BN',
    'block_14_expand',
    'block_14_expand_BN',
    'block_14_expand_relu',
    'block_14_depthwise',
    'block_14_depthwise_BN',
    'block_14_depthwise_relu',
    'block_14_project',
    'block_14_project_BN',
    'block_14_add',
    'block_15_expand',
    'block_15_expand_BN',
    'block_15_expand_relu',
    'block_15_depthwise',
    'block_15_depthwise_BN',
    'block_15_depthwise_relu',
    'block_15_project',
    'block_15_project_BN',
    'block_15_add',
    'block_16_expand',
    'block_16_expand_BN',
    'block_16_expand_relu',
    'block_16_depthwise',
    'block_16_depthwise_BN',
    'block_16_depthwise_relu',
    'block_16_project',
    'block_16_project_BN']
}