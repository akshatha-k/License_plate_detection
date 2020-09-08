# License_plate_detection

This repo builds on the work of [alpr-constrained](https://github.com/sergiomsilva/alpr-unconstrained) , [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)

## Commands

#### Annotation
Runs the annotate tool. \
Use this to create annotations for the license plate images.

```
python3 annotation-tool.py <max_image_height> <max_image_width> <List of image files separated by a space>
```
Pass every single image file name as an argument to the script

#### Arguments
```
- model : Path to previous model
- name : Model name
- lp_model : Pre-trained model pat
- optimizer : Optimizer (default = Adam)
- learning_rate : Optimizer (default = 0.01)
- batch_size : Mini-batch size (default = 32)
- image_size : Image size
- epochs : Number of training epochs
- num_augs : Total number of images after random augmentations
- use_colab : Use google colab
- resume : Resume from ckpt
- prune_model : Whether to prune the model or not
- initial_sparsity : Initial sparsity while pruning
- final_sparsity : Final sparsity while pruning
- begin_step : Start pruning point
- end_step : End pruning point
- lr_steps :  Cycle step for cyclical LR
- lr_schedule : LR scheduler to use cyclic,step
- max_lr : Max Learning rate
- min_lr : Min Learning rate
```
#### Create the Model 
```
cd License_plate_detection/ && python create-model.py <arguments>
```
#### Train the model
```
cd License_plate_detection/ && python train-detector.py <arguments>
```

#### Baseline Inference
```
cd License_plate_detection/ && python lp-detection.py <arguments>
```
#### Pruning support
Use argument prune_model True and set the other parameters according to the args.py
```
d License_plate_detection/ && python lp-detection.py --prune_model True
```

#### Create TFLite model
```
cd License_plate_detection && python create_tflite.py 
```
#### TFLite inference
```
cd License_plate_detection && python lp-tflite-detection.py
```