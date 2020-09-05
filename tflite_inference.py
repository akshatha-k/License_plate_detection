import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf

from args import get_args

args = get_args()
if args.use_colab:
    from google.colab import drive

    drive.mount('/content/gdrive')
    OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}'.format(args.image_size, args.initial_sparsity,
                                                                args.final_sparsity)
    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    test_dir = '/content/gdrive/My Drive/lpd/test_images'
    output_dir = '{}/{}'.format(OUTPUT_DIR, 'output_dir')
    tflite_path = '{}/{}.tflite'.format(OUTPUT_DIR, args.model)

IMAGE_SIZE = args.image_size
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample_dir = test_dir
onlyfiles = ["{}/{}".format(sample_dir, f) for f in listdir(sample_dir) if isfile(join(sample_dir, f))]
for i, img_path in enumerate(onlyfiles):
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    input_data = image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
