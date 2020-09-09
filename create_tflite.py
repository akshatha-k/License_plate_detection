import os

import tensorflow as tf

from args import get_args
from src.utils import get_logger, setup_dirs

setup_dirs()
logger = get_logger("create-tflite")
args = get_args()
if args.use_colab:
    from google.colab import drive

    drive.mount('/content/gdrive')
    OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}_{}_{}'.format(args.image_size, args.epochs,
                                                                args.prune_model,
                                                                   args.initial_sparsity,
                                                                   args.final_sparsity)
    if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    model_path_final = '{}/{}_trained'.format(OUTPUT_DIR, args.model)
    pruned_path_final = '{}/{}_pruned'.format(OUTPUT_DIR, args.model)
    model_name = '{}/{}'.format(model_path_final, args.model)
    tflite_path = '{}/{}.tflite'.format(OUTPUT_DIR, args.model)
    pruned_tflite_path = '{}/{}_pruned.tflite'.format(OUTPUT_DIR, args.model)

IMAGE_SIZE = args.image_size
model = tf.saved_model.load(model_path_final)
concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, IMAGE_SIZE, IMAGE_SIZE, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter = tf.lite.TFLiteConverter.from_saved_model("/home/akshatha/LP_detect/License_plate_detection/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This ensures that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# These set the input and output tensors to uint8
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# And this sets the representative dataset so we can quantize the activations
# converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

open(tflite_path, "wb").write(tflite_model)

if args.prune_model:
    model = tf.saved_model.load(pruned_path_final)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, IMAGE_SIZE, IMAGE_SIZE, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # converter = tf.lite.TFLiteConverter.from_saved_model("/home/akshatha/LP_detect/License_plate_detection/saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This ensures that if any ops can't be quantized, the converter throws an error
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # These set the input and output tensors to uint8
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # And this sets the representative dataset so we can quantize the activations
    # converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    open(pruned_tflite_path, "wb").write(tflite_model)
