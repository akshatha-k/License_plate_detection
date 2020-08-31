import tensorflow as tf

IMAGE_SIZE=208
sample_dir="/home/akshatha/LP_detect/License_plate_detection/samples/test"
model = tf.saved_model.load("/home/akshatha/LP_detect/License_plate_detection/saved_model")
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 208, 208, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# # A generator that provides a representative dataset
# def representative_data_gen():
#   dataset_list = tf.data.Dataset.list_files(sample_dir + '/*')
#   for i in range(8):
#     image = next(iter(dataset_list))
#     image = tf.io.read_file(image)
#     image = tf.io.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image = tf.cast(image / 255., tf.float32)
#     image = tf.expand_dims(image, 0)
#     yield [image]

# converter = tf.lite.TFLiteConverter.from_saved_model("/home/akshatha/LP_detect/License_plate_detection/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This ensures that if any ops can't be quantized, the converter throws an error
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# These set the input and output tensors to uint8
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# And this sets the representative dataset so we can quantize the activations
#converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)