#!/usr/bin/python3
# -*- coding: utf-8 -*- 

# May 25th, 2022

#https://neptune.ai/blog/how-to-build-a-light-weight-image-classifier-in-tensorflow-keras
#https://towardsdatascience.com/model-compression-a-look-into-reducing-model-size-8251683c338e

# SUMMARY of keras models in tf.keras.applications
#https://keras.io/api/applications/#usage-examples-for-image-classification-models
# mobilenet v2 is smallest with 14 mb and 3,5M params


# either create new model with conv + custo classifier (can use data augment, which is needed for small custom data set)
# or extract feature from conv only and run as input to another small model

# In general, it's a good practice to develop models that take raw data as input, 
# as opposed to models that take already-preprocessed data. do preprocessing in the model itself
# resize in data pipeline. rescale in model tf.keras.layers.Rescaling

fine_tune_epochs = 7 # train upper layers
initial_epochs = 8  # with frozen model. train classifier

name_of_model = 'mobilenetv2_224' # print in benchmark tables
# is using model maker, must match original tflite input size, otherwize error at inference (why not on training ?)
IMG_SIZE = (96, 96)  # # used to build dataset. dataset elements are resized to this, from what's already in classes dir
IMG_SHAPE = IMG_SIZE + (3,) # use as input to mobilenet
#IMG_SHAPE = (160,160,3) # default mobilenet is (224,224,3)

from dataclasses import dataclass
from importlib_metadata import metadata
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorboard
#from sklearn.metrics import confusion_matrix # https://stackoverflow.com/questions/59474533/modulenotfounderror-no-module-named-numpy-testing-nosetester
import tensorflow as tf
import sys
import pathlib
import glob
from tabulate import tabulate
from prettytable import PrettyTable

from PIL import Image

import logging

from tensorflow_hub import image_embedding_column
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

print('transfert learning')

# pip install tabulate prettytable pydot cf
# install windows graphviz https://graphviz.org/download/ or apt install

print(tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE

#print('GPU available ', tf.test.is_gpu_available( cuda_only=True, min_cuda_compute_capability=(3,0)))
print('GPU available ', tf.test.is_gpu_available()) # compute capability 5

print('GPU ', tf.config.list_physical_devices('GPU'))
print('built with CUDA ', tf.test.is_built_with_cuda)

# pabou.py is one level up
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..') # this is in above working dir
#print ('sys path used to import modules: %s\n' % sys.path)
try:
    print ('importing pabou helper')
    import pabou # various helpers module, one dir up
    import arg # arg parsing
    import lite_inference # single inference and benchmark
    # pip install prettytable
    print('pabou helper path: ', pabou.__file__)
    #print(pabou.__dict__)
except:
    print('cannot import pabou. check it runs standalone. syntax error in  will fail the import')
    exit(1)

arg = arg.parse_arg()
app = arg['app']
models_dir = pabou.models_dir
label_txt = pabou.label_txt
label_file = app + label_txt # name of label file 
label_file = os.path.join (models_dir, label_file) # label file path. to be created. expected by TFlite metadata. also for micropython


#######################
# apply argmax on ONE softmax, or array of softmax, comming frol predict on array of inputs
# ONLY woks for dim = 1 or 2.
def func(a):
  #a = tf.nn.sigmoid(a)
  # By default, the index is into the flattened array, otherwise along the specified axis.
  # either one array of softmax, or multiples
  if len(a.shape) > 1:
    return(np.argmax(a, axis=1))
  else:
    return(np.argmax(a))

#####################################################
# returns train dir and validation dir
# classes sub labels
#####################################################



# all images, with classes sub labels
def get_input_dir():
  c = os.getcwd()
  return( os.path.join(c, '..', 'data_augmentation', 'datasets'))

###################################################
# create dataset, yield elements
# dataset has element single pic , created with BATCH_SIZE = 1
# dataset_batch has element batch of 32 pic
###################################################

def get_dataset(classes_dir): 
  global softmax_size

  BATCH_SIZE = 1  # one element = one batch of pic.  
  # batch size is a feature from image_dataset_from_directory. 
  # creating dataset with range() has "batch" = 1. 

  # can create batch with .batch(), BUT ADD EXTRA DIM

  # use .take(n) to create new dataset with a n element (ie batch). returns TakeDataset. 
  # use for loop on dataset to get element one by one (element = batch). until end of data set

  # batch = 1 allows ONE for loop to access single pic sequentially 
  # vs batch = 16, for loop return 12 batches, each with 16 pict (array of tensors)
  
  # can use next() to get a single batch. returns ndarray WITHOUT for loop, ie one statement
  # image_batch, label_batch = test_dataset.as_numpy_iterator().next()
  # image_batch, label_batch = next(iter(train_dataset)) 

  # get array of element WITHOUT for loop with batch = 1 . see representative gen !!!!!
  #bs = dataset.batch(10).take(1) # new dataset is take 1 element. element is batch of 10 pic
  #image_batch, label_batch = bs.as_numpy_iterator().next() # array of dim 10, 1, 160, 160, 3
  #image_batch = np.squeeze(image_batch, axis = 1)

  # tensor.numpy() convert to numpy array

  # Generates a tf.data.Dataset from image files in a directory.
  # dir/class_a/image.jpg
  # labels cans be encoded. inferred = from dir structure
  # cannot use 1 for split
  # ENTIERE train and validation dataset can be passed to fit
  # pixels are float32 0 to 255

  # good practice to develop models that take raw data as input, vs models that take already-preprocessed data

  # could do train/validation split here ie use single directory, or before in flow_from_directory and have separate directory
  
  """
  data augmentation uses a keras python iterator to create augmented pic on disk
  bla = ImageDataGenerator() and iterator = bla.flow_from_directory()
  
  can create tf dataset from this iterator
  ds = tf.data.Dataset.from_generator( lambda: iterator,  )
  
  here we create IMAGE dataset directly from train and validation directory
  """
  # https://www.tensorflow.org/tutorials/load_data/images
  train_dataset = tf.keras.utils.image_dataset_from_directory(classes_dir, 
  labels='inferred', label_mode='int', color_mode='rgb', 
  shuffle=True, validation_split = None, subset = None, seed=1000 , class_names = None, 
  batch_size=BATCH_SIZE, image_size=IMG_SIZE)

  # training much slower by using dataset with batch size = 1.
  train_dataset_batch = tf.keras.utils.image_dataset_from_directory(classes_dir, 
  labels='inferred', label_mode='int', color_mode='rgb', 
  shuffle=True, validation_split = None, subset = None, seed=1000 , class_names = None, 
  batch_size=32, image_size=IMG_SIZE)

  # class names: explicit list of classes matching directory. to control order. otherwize os.walk, alphanumeric

  # Found x files belonging to y classes.
  # if validation split = 0.2 and subset = 'training' , 2000 => 1600
  # label as int: SparseCategoricalCrossentropy. categogical = one hot. binary 2 float32 scalars
  # cardinality returns scalar tensor in number of BATCH, ie 50   32x50 = 1600 tf.Tensor(50, shape=()
  # one dataset element is one batch
  
  # dataset characteristics
  print('train dataset cardinality in batch', train_dataset.cardinality() , train_dataset.cardinality().numpy(), "%d" %train_dataset.cardinality())

  # another notation
  assert (tf.data.experimental.cardinality(train_dataset) == train_dataset.cardinality())
  assert (train_dataset.cardinality() == len(list(train_dataset)))

  print('train dataset len ', len(list(train_dataset))) # 234
  print('train dataset_batch len ', len(list(train_dataset_batch))) # 8

  #Found 2000 files belonging to 2 classes.
  #Using 1600 files for training. if validation_split = 0.2, subset = "training", seed=1000

  # element characteristics
  print('train dataset element spec ', train_dataset.element_spec) # tuple of Tensorspec
  # TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32, name=None)
  # TensorSpec(shape=(None,), dtype=tf.int32, name=None)

  for img, label in train_dataset: # get element one by one (element = batch)
    print('single element: ', img.numpy().shape, label.numpy().shape) # (1, 160, 160, 3) (1,)
    break

  # arbitrary processing as part of input pipeline, can parralelize preprocessing on CPU cores 
  #ds.map(pre processing function, num_parallel_caller = 4)
  
  # using batch add one extra dim
  bs =train_dataset.batch(10).take(1) # create extra dim
  for image_batch, label_batch in bs: # run ONCE because take(1). EXTRA dim 
    #print(image_batch.shape) # shape=(10, 1, 160, 160, 3), dtype=float32)
    #print(label_batch.shape) # shape=(10, 1), dtype=int32)
    pass

  # another way 
  image_batch, label_batch = bs.as_numpy_iterator().next()
  #print(image_batch.shape) # (10, 1, 160, 160, 3)
  #print(label_batch.shape) # 
  
  
  # remove extra dim 
  image_batch = np.squeeze(image_batch, axis = 1)
  label_batch = np.squeeze(label_batch, axis = 1)
  #print(image_batch.shape) # (10, 160, 160, 3)
  #print(label_batch.shape) # (10,)


  # classes
  class_names = train_dataset.class_names # class name  ['cats', 'dogs']
  print('class name ', class_names) # unless class_names is set, order is alphanumeric
  # class_name[label as int] to get string

  # used when creating model
  softmax_size = len(class_names)

  # create train, validation and test set
  # may result in empty test set if set has few pic and BATCH size high
  elements = train_dataset.cardinality() # number of pic, as batch size is 1

  # size of test and validation
  tmp = elements // 6

  # use take and skip to create sub dataset 
  test_dataset = train_dataset.take(tmp)
  validation_dataset = train_dataset.take(tmp)
  train_dataset = train_dataset.skip(2*tmp)

  print('Number of train (1) : %d' % train_dataset.cardinality().numpy())
  print('Number of test (1): %d' % test_dataset.cardinality().numpy())
  print('Number of validation(1): %d' % validation_dataset.cardinality().numpy())

  elements = train_dataset_batch.cardinality() # in batches
  tmp = elements // 6

  test_dataset_batch = train_dataset_batch.take(tmp)
  validation_dataset_batch = train_dataset_batch.take(tmp)
  test_dataset_batch = train_dataset_batch.skip(2*tmp)

  print('Number of train batches(32): %d' % train_dataset_batch.cardinality().numpy())
  print('Number of test batches(32): %d' % test_dataset_batch.cardinality().numpy())
  print('Number of validation batches(32): %d' % validation_dataset_batch.cardinality().numpy())

  # could also create a test set by using validation split
  """
  test_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, 
  labels='inferred', label_mode='int',
  shuffle=True, validation_split = 0.2, subset = "validation", seed=1000 , 
  batch_size=BATCH_SIZE, image_size=IMG_SIZE)
  """
  # take Creates a Dataset with at most count elements from this dataset.
  # alternative: image_batch, label_batch = next(iter(train_dataset)) 

  ds = train_dataset.take(9) 
  # still a dataset TakeDataset. len(list(ds)) is 9
  # 9 time (1, 160, 160, 3)
  #one dataset element:  (TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32,  name=None))
  #9  elemnts

  ds = train_dataset.batch(9).take(1) 
  # !!!!!!! .batch() add extra dim and convert 9 elements into 1, or 9 into 3 with batch(3)
  # shape=(None, None, 160, 160, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32,  name=None))
  #1  elemnts
  #1 time (9, 1, 160, 160, 3)
 
  ############################
  # show some input images
  # user plt.imshow() for subplot
  # can use PIL.Image.open(str).show()
  # imshow for (M, N, 3): an image with RGB values (0-1 float or 0-255 int). 
  # convert 255.0 float32 with .astype("uint8")  or /255 
  # check dtype and range for dataset images (float32 and 0.0 to 255.0)
  ############################
  # need enough image to build the 3 x 3 figure

  if not arg['headless']: 
    # dataset has 1 pic per element
    pabou.plot_9_images_1(plt, train_dataset.shuffle(1000), (6,6), 'some input pictures')
    #plt.show()

  #Use buffered prefetching to load images from disk without having I/O become blocking. (https://www.tensorflow.org/guide/data_performance) guide.
  AUTOTUNE = tf.data.AUTOTUNE
  # Creates a Dataset that prefetches elements from this dataset.
  # Most dataset input pipelines should end with a call to prefetch. 
  # This allows later elements to be prepared while the current element is being processed. 
  # latency and throughput vs additional memory


  # also shuffle. 1000 = buffer size. shuffle within this buffer. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
  
  train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  validation_dataset = validation_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  test_dataset = test_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)

  # PrefetchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32

  train_dataset_batch = train_dataset_batch.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  validation_dataset_batch = validation_dataset_batch.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  test_dataset_batch = test_dataset_batch.shuffle(1000).prefetch(buffer_size=AUTOTUNE)

  return(train_dataset, validation_dataset, test_dataset, class_names, train_dataset_batch, validation_dataset_batch, test_dataset_batch)
 

####################################################################
# TRANSFERT
####################################################################

def transfert_learning(train_dataset, validation_dataset, test_dataset, 
  class_names, 
  train_dataset_batch, validation_dataset_batch, test_dataset_batch):
  

  ##############################################
  # data augmentation 
  # can be included in model definition as 'pre processing' (only used for training)
  # or applied to dataset as ds = model(ds)
  # technique to increase the diversity of your training set by applying random (but realistic) transformations, such as image rotation.
  
  # this is on top of syntetic data created to extend training set
  ##############################################

  # data augmentation
  #(https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). 

 
  # data augmentation: can use either built in keras layers, or tf.image

  # USING tf.image 
  # .map on dataset 
  # https://www.tensorflow.org/tutorials/images/data_augmentation

  print('apply data augmentation using tf image')
  f = pabou.do_image_augmentation

  # create a dataset with augmented images, can use for training
  new_train_dataset_batch = train_dataset_batch.shuffle(1000).map(f, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)


  # USING built in keras layers:

  # https://www.tensorflow.org/guide/keras/preprocessing_layers 
  # do not use layers which change image w,h , eg crop, or which are not adapted to face images (eg flip)
  #tf.keras.layers.RandomFlip('horizontal_and_vertical'),
  #tf.keras.layers.RandomBrigthness() only in pip nigthly

  # zoom vertical zoom height_factor=(0.2, 0.3) result in an output zoomed out by a random amount in the range [+20%, +30%]. height_factor=(-0.3, -0.2) result in an output zoomed in by a random amount in the range [+20%, +30%].
  #  width_factor Defaults to None, i.e., zooming vertical and horizontal directions by preserving the aspect ratio.
  # rotation factor=(-0.2, 0.3) results in an output rotation by a random amount in the range [-20% * 2pi, 30% * 2pi]. factor=0.2 results in an output rotating by a random amount in the range [-20% * 2pi, 20% * 2pi].
  # contast For each channel, this layer computes the mean of the image pixels in the channel and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean.
  # factor between 0 and 1
  #tf.keras.layers.RandomZoom((0.1, 0.4), width_factor=None, fill_mode= 'reflect'),

  data_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation((-0.1, 0,2)),
    tf.keras.layers.RandomContrast(0.2),
  ])

  # Data augmentation is inactive at test time so input images will only be augmented during calls to Model.fit (not Model.evaluate or Model.predict).

  # keras layers can be applied on images and dataset.map()

  # rescaling:
  # decide where to put rescaling. in pictures, in dataset, in model as pre processing, or as a specific mobilenet preprocessing function
  
  #Option 1: Make the preprocessing layers part of your model
  # Data augmentation will run on-device, synchronously with the rest of your layers, and benefit from GPU acceleration.
  # When you export your model using model.save, the preprocessing layers will be saved along with the rest of your model. If you later deploy this model, it will automatically standardize images (according to the configuration of your layers). This can save you from the effort of having to reimplement that logic server-side.
  
  #Option 2: Apply the preprocessing layers to your dataset
  # Data augmentation will happen asynchronously on the CPU, and is non-blocking. You can overlap the training of your model on the GPU with data preprocessing, using Dataset.prefetch, shown below.
  # In this case the preprocessing layers will not be exported with the model when you call Model.save. You will need to attach them to your model before saving it or reimplement them server-side. After training, you can attach the preprocessing layers before export.
  #  ds = train_ds.map( lambda x, y: (resize_and_rescale(x, training=True), y), num_parallel_calls=AUTOTUNE)
  #  ds.prefetch(buffer_size=AUTOTUNE)
  
  # !!!! do rescaling in model x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
  # so model expects float32 0.0 to 255.0

  resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.Rescaling(1./255)
  ]) # not used, use tf.keras.applications.mobilenet_v2.preprocess_input(x)


   
  print('show some data augmentation')
  # or use image_batch, label_batch = next(iter(train_dataset)) for a single image
  # assume batch = 1

  if not arg['headless']:
    # use dataset with element = 1 pic

    def f(image , label):
      image = data_augmentation_layers(image)
      return (image, label)

    #  apply keras data augmentation layers to dataset
    #ds = train_dataset.map(data_augmentation_layers) # <MapDataset element_spec=TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None)>
    # this worked, but problem later when iterating on ds

    ds = train_dataset.shuffle(1000).map(f,  num_parallel_calls=AUTOTUNE) 

    pabou.plot_9_images_1(plt, ds, (6,6), 'data augmentation', class_list = class_names)

    plt.savefig('data_augmentation.jpg')
    
    #plt.show()
      
  # data augmentation layer is included in model. will be used at training only

  ###################################################################
  # mobilenet without classifier
  # output is called bottleneck
  # 5x5x1280
  # will be flattened (global average pooling 2D), and ru thru dense(4)
  ###################################################################

  print('create mobilenet WITHOUT classifier')
 
  # MobileNetV2` expects pixel values in `[-1, 1]`,  images are in `[0, 255]`
  # tf.keras.applications   all models

  # DO NOT include top classifier
  # mobilenet 0 to 1 float32

  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,  include_top=False, weights='imagenet')
  
  # poor accuracy cf
  #base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,  include_top=False, weights='imagenet', minimalistic = False)

  #WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.

  #input_1 (None, 160, 160, 3) <dtype: 'float32'> (None, 160, 160, 3) <dtype: 'float32'>
  #out_relu (None, 5, 5, 1280) <dtype: 'float32'> (None, 5, 5, 1280) <dtype: 'float32'>
  
  #base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE, 
  #include_top=False, weights='imagenet') # test 0.44

  #base_model.input <KerasTensor: shape=(None, 160, 160, 3) dtype=float32 (created by layer 'input_1')>
  #base_model.inputs [<KerasTensor: shape=...input_1')>]  
  #base_model.inputs[0] <KerasTensor: shape=(None, 160, 160, 3) dtype=float32 (created by layer 'input_1')>

  assert (base_model.input.shape == base_model.inputs[0].shape) # inputs is a list. here one input only

  # get some layer charasteristics
  # see layers
  print('model has %d layers'  %len(base_model.layers))
  for i in [0,1,-2,-1]:
    try:
      print(base_model.layers[i].name, base_model.layers[i].input.shape, base_model.layers[i].input.dtype ,  base_model.layers[i].output.shape, base_model.layers[i].output.dtype)
    except: # eg in mobilenet V3, the input to last layer is not a tensor, but a list of tensors
      pass 
  print('last layer , ie bottleneck ', base_model.get_layer(name=None, index = -1).output ) 
 
  # run image thru network. output is bootleneck
  # get a batch of image
  image_batch, label_batch = next(iter(train_dataset))  # TensorShape([32, 160, 160, 3]) or [1,160 ..]
  # alternate way to: for (img, lab) in train_dataset.take(1):  

  # can predict with calling entire model, but also works for calling individual layers
  # feature = bootleneck . layer before flatten(aka top) is bottleneck
  # run thru frozen, headless model. output bootleneck
  bottleneck_feature_batch = base_model(image_batch) 
  # call model with input = batch of 32. returns feature from bottleneck layer (eager tensor)
  print('bootleneck feature from model(): ', bottleneck_feature_batch.shape, type(bottleneck_feature_batch.shape)) #(32, 5, 5, 1280) <class 'tensorflow.python.framework.ops.EagerTensor'>
  # 1280 maps of 5x5 

  # another way to get output from model, using simply predict
  # model() returns tensor, model.predict return ndarray
  tmp = base_model.predict(image_batch)
  print('bottleneck feature from model.predict', tmp.shape, type(tmp))

  assert (bottleneck_feature_batch.shape == tmp.shape)

  # not the same array. likely because not the same input 
  #assert(np.array_equal(bottleneck_feature_batch.numpy(), tmp ))

  # freeze model
  # trainable: updated via gradiant descend. non trainable updated in forward pass
  # In general, all weights are trainable weights. only batch norm has non trainable

  print('layers: %d, trainable: %d %d. non trainable: %d'  %( len(base_model.weights), 
  len(base_model.trainable_weights), len(base_model.trainable_variables), len(base_model.non_trainable_weights)))

  # [<tf.Variable 'Conv1/...=float32)>, <tf.Variable 'bn_Con...=float32)>, 
  # weight, trainable before freezing 260 156 156    not the number of variable , but of tf variable

  # could freeze based on name
  print('freeze model')
  base_model.trainable = False # for all layers
  for layers in base_model.layers:
    #print(layers.name, layers.trainable, end = ' ')
    layers.trainable = False # not needed

  print('layers: %d, trainable: %d %d. non trainable: %d'  %( len(base_model.weights), 
  len(base_model.trainable_weights), len(base_model.trainable_variables), len(base_model.non_trainable_weights)))
  #trainable before freezing 260 0 0

  """
  Important note about BatchNormalization layers
  Many models contain `tf.keras.layers.BatchNormalization` layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial. 
  When you set `layer.trainable = False`, the `BatchNormalization` layer will run in inference mode, and will not update its mean and variance statistics. 
  When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, you should keep the BatchNormalization layers in inference mode by passing `training = False` when calling the base model. Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.
  For more details, see the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).
  """

  # NO classifier out_relu (ReLU)                (None, 5, 5, 1280)   Non-trainable params: 2,257,984

  print('create custom classifier')

  # global average pooling 2D is used intead of flatten to reduce dimension
  # The 2D Global average pooling block takes a tensor of size (input width) x (input height) x (input channels) and computes the average value of all values across the entire (input width) x (input height) matrix for each of the (input channels).
  # ie goes from W x H x depth to 1 x 1 x Depth  , ie 5x5 x 1280 to 1280
  # average each W x H into one scalar
  # reduce dim by 2
  # flatten to 1280
  # alternative to the Flattening block after the last pooling block of your convolutional neural network. 
  # Using 2D Global average pooling block can replace the fully connected blocks of your CNN.

  # convert the features to  a single 1280-element vector per image.

  # output from frozen (bottleneck) ie 5x5X1280 fed into 2D average (flatten 1280) and dense (4)
  
  pooling_2D = tf.keras.layers.GlobalAveragePooling2D() 
  feature_batch_average = pooling_2D(bottleneck_feature_batch)
  print('bottleneck thru global average pooling2D is flattening', bottleneck_feature_batch.shape, feature_batch_average.shape) 
  #(16, 5, 5, 1280) to (16, 1280)
 
  #You don't need an activation function here because this prediction will be treated as a `logit`, or a raw prediction value. 
  
  # run pooling thru dense
  prediction_layer = tf.keras.layers.Dense(softmax_size)
  prediction_batch = prediction_layer(feature_batch_average) 
  print('run pooling thru dense', prediction_batch.shape) 
  #run pooling thru dense (16, 4)


  ##############################################################################
  # keras layer data augmentation
  # mobilenet preprocess (rescale)
  # base , frozen mobilenet, without classifier
  # flatten, dropout, dense
  ##############################################################################


  print('add preprocessing and custom classifier to frozen base mobilenet')
  
  #Build a model by chaining together the data augmentation, rescaling, `base_model` and  feature extractor layers using the (https://www.tensorflow.org/guide/keras/functional). 
  #As previously mentioned, use `training=False` as our model contains a `BatchNormalization` layer.

  # MobileNetV2` expects pixel values in `[-1, 1]`,  images are in `[0.0, 255.0]`
  # to 0 255 to -1 +1 could use layer or preprocess_inout
  rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

  # define new model: augment/rescale + frozen + pooling/dense 
  # model INCLUDES base_model

  inputs = tf.keras.Input(shape=IMG_SHAPE)

  x = data_augmentation_layers(inputs)

  x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # use builtin to rescale 0.0 255.0 to 0 1

  # this is mobilenet
  x = base_model(x, training=False) # see batch norm 
  # x is bottleneck layer

  x = tf.keras.layers.GlobalAveragePooling2D() (x)
  x = tf.keras.layers.Dropout(0.2)(x)

  print('softmax size ', softmax_size)

  #outputs = tf.keras.layers.Dense(1, activation = None) (x) # binary 
  # keeps logits instead of sigmoid. numerical stability ?  recommended for binary classification.
  # activation = None (vs sigmoid) means NO softmax and from_logits=True in compile
  outputs = tf.keras.layers.Dense(softmax_size, activation = 'softmax') (x) # 3 classes

  model = tf.keras.Model(inputs, outputs)

  # get layer charasteristics
  print('First and last 2 layers')
  for i in [0,1,-2,-1]:
    print(model.layers[i].name, model.layers[i].input.shape, model.layers[i].input.dtype , model.layers[i].output.shape,model.layers[i].output.dtype)

  #dense_1 KerasTensor(type_spec=TensorSpec(shape=(None, 1280), dtype=tf.float32, name=None), name='dropout/Identity:0', description="created by layer 'dropout'") KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None), 
  #name='dense_1/BiasAdd:0', description="created by layer 'dense_1'")

  #The from_logits=True attribute inform the loss function that the output values generated by the model are not normalized, a.k.a. logits. In other words, the softmax function has not been applied on them to produce a probability distribution. 
  #Therefore, the output layer in this case does not have a softmax activation function:
  # logits Whether to interpret y_pred as a tensor of logit values. 
  # By default, we assume that y_pred contains probabilities (i.e., values in [0, 1]).
  # numerical stability ?
  # Recommended Usage: (set from_logits=True)
  #Use this cross-entropy loss for binary (0 or 1) classification applications. The loss function requires the following inputs:
  #y_true (true label): This is either 0 or 1.
  #y_pred (predicted value): This is the model's prediction, i.e, a single floating-point value which either represents a logit, (i.e, value in [-inf, inf] when from_logits=True) or a probability (i.e, value in [0., 1.] when from_logits=False).

  base_learning_rate = 0.0001
  #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True) # multi label , label as one hot
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # multi label as int. SOFTMAX
  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
  loss=loss, metrics=['accuracy'])

  #dense_1 (Dense)             (None, 1)                 1281 for binary
  #dense_1 (Dense)             (None, 3)                 3843
  #Trainable params: 3,843
  #Non-trainable params: 2,257,984

  pabou.see_model_info(model, 'custom classifier.png')

  """
  The 2.5 million parameters in MobileNet are frozen, but there are 1.2 thousand _trainable_ parameters 
  in the Dense layer. These are divided between two `tf.Variable` objects, the weights and biases.
  Total params: 2,259,265
  Trainable params: 1,281
  Non-trainable params: 2,257,984
  """
  print('trainable layers', len(model.trainable_variables))

  # accuracy before training classifier
  loss0, accuracy0 = model.evaluate(validation_dataset)
  print("initial accuracy (val): %0.2f" %accuracy0) 

  loss0, accuracy0 = model.evaluate(test_dataset)
  print("initial accuracy(test): %0.2f" %accuracy0)

  print('train model using dataset. faster to use dataset with batched images')

  # with train_dataset (ie element = single pic) 8 sec / epoch after epoch 1
  # with train_dataset_batch (elemnt 32 pic) 1 sec /epoch

  what_to_monitor = 'val_accuracy'
  root = os.getcwd()
  print("starting dir: ", root)
  checkpoint_path = os.path.join( root, 'checkpoint', 'cp-{epoch:03d}.ckpt')
  print('save checkpoint (formatted) with checkpoint path: %s' %(checkpoint_path))
  tensorboard_log_dir = os.path.join(root, 'tensorboard_log_dir')

  callbacks_list = [
            
  # stop when val_accuracy not longer improving, no longer means 1e-2 for 4 epochs. can use baseline
  tf.keras.callbacks.EarlyStopping(monitor=what_to_monitor, mode= 'auto' , min_delta=1e-2, restore_best_weights=True, patience=8, verbose=1),
  
  # factor: factor by which the learning rate will be reduced. new_lr = lr * factor 
  # min_lr: lower bound on the learning rate.
  tf.keras.callbacks.ReduceLROnPlateau(monitor=what_to_monitor,factor=0.1,patience=8, min_lr=0.001),
  
  #tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=what_to_monitor, mode = 'auto', save_best_only=True, verbose=1, save_weights_only = True, save_freq = 'epoch', period=1),
  
  #Epoch 00001: val_loss improved from inf to 0.25330, saving model to mymodel_1.h5
  #stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain: * One or more shards that contain your model's weights. * An index file that indicates which weights are stored in a which shard.
  #If you are only training a model on a single machine, you'll have one shard with the suffix: .data-00000-of-00001
  # checkpoint dir created if needed

  tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir , update_freq='epoch', histogram_freq=1, write_graph=False, write_images = True)

  # histogram frequency in epoch. at which to compute activation and weight histograms for the layers
  # write_images: whether to write model weights to visualize as image in TensorBoard.
  # update_freq: 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch.
  # write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
  ]

  # callbacks seems to slow training. likely tensorboard
  callbacks_list = []

  # train on train_dataset_batch
  history = model.fit(train_dataset_batch, epochs=initial_epochs, validation_data=validation_dataset_batch, callbacks=callbacks_list, verbose = 1)

  # history <keras.callbacks.History object at 0x000002349B278C48> attributes: params , history , epochs
  
  # history.params {'verbose': 1, 'epochs': 10, 'steps': 50}
  # history.epoch [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  # history.history is dict. accuracy list of floats

  history_dict = history.history

  print('history_dict keys ', history_dict.keys())
  #history.history.keys() dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
  print('len of metric/epochs ', len(history_dict['accuracy']))

  # single_output = True: model with a single output
  # plot accuracy (training validation) and loss (training validation) on two sub plot
  # pass history.history, ie dict

  pabou.see_training_result(history_dict, accuracy0, 'train_result_frozen.png', single_output = True, fig_title ='frozen' )

  """
  validation metrics are clearly better than the training metrics, 
  the main factor is because layers like `tf.keras.layers.BatchNormalization` and `tf.keras.layers.Dropout` 
  affect accuracy during training. They are turned off when calculating validation loss.
  To a lesser extent, it is also because training metrics report the average for an epoch, 
  while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.
  """


  ########################################
  # create fine tune model
  print ('fine tune, after frozen model trained to convergence')
  ########################################
  """
  train (or "fine-tune") the weights of the top layers of the pre-trained 
  only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. 
  If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.
  try to fine-tune a small number of top layers rather than the whole MobileNet model. 

  unfreeze the `base_model`, set the bottom layers to be un-trainable. 
  recompile the model (necessary for these changes to take effect), and resume training.
  """

  # do not modify model architecture, just unfreeze more layers
  # It is critical to only do this step after the model with frozen layers has been trained to convergence. 
  base_model.trainable = True

  # Let's take a look to see how many layers are in the base model
  print("base model layers: ", len(base_model.layers))

  # Fine-tune from this layer onwards
  fine_tune_at = 100

  print('base model layers: total %d, trainable: %d %d. non trainable: %d'  %( len(base_model.weights), 
  len(base_model.trainable_weights), len(base_model.trainable_variables), len(base_model.non_trainable_weights)))

  # Freeze all the layers before the `fine_tune_at` layer
  # could freeze based on name
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

  print('after fine tuning, ie de freezing some')
  print('base model layers: %d, trainable: %d %d. non trainable: %d'  %( len(base_model.weights), 
  len(base_model.trainable_weights), len(base_model.trainable_variables), len(base_model.non_trainable_weights)))

  ########################################
  # compile, fit, evaluate and save fine tune
  ########################################

  # use a lower learning rate at this stage. Otherwise, your model could overfit very quickly.
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
  # sparse categorical = multi label as int. logits = False = last dense has activation = softmax
  
  model.compile(loss=loss,
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
  metrics=['accuracy'])

  model.summary()
  #Total params: 2,259,265
  #Trainable params: 1,862,721
  #Non-trainable params: 396,544

  pabou.see_model_info(model, 'fine tune.png')

  total_epochs =  initial_epochs + fine_tune_epochs

  # resume training from last epoch
  #  0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases,
  # initial epoch Integer. Epoch at which to start training (useful for resuming a previous training run). 
  # model has not forgotten previous training. resume at epoch. epoch starts at zero. epoch[-1] or len(history.history['accuracy'] -1
  #It's also critical to use a very low learning rate at this stage, because you are training a much larger model than in the first round of training, on a dataset that is typically very small

  # with train dataset batch size = 1, training much slower

  """
  fit can take as X
  A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
  A tf.data dataset.
  A generator or keras.utils.Sequence instance.
  """

  history_fine = model.fit(train_dataset_batch, epochs=total_epochs, initial_epoch=history.epoch[-1], 
  validation_data=validation_dataset_batch , callbacks=callbacks_list, verbose = 1)

  #You may also get some overfitting as the new training set is relatively small and similar to the original MobileNetV2 datasets.

  # add metrics from new epoch . concatenate list 
  # frozen training in history_dict
  # finetuning training in history_fine_dict
  history_fine_dict = history_fine.history

  total_history = {}

  ac = history_dict['accuracy'] + history_fine_dict['accuracy']
  vac = history_dict['val_accuracy'] + history_fine_dict['val_accuracy']
  lo = history_dict['loss'] + history_fine_dict['loss']
  vlo = history_dict['val_loss'] + history_fine_dict['val_loss']

  total_history['accuracy'] = ac
  total_history['val_accuracy'] = vac
  total_history['loss'] = lo
  total_history['val_loss'] = vlo
 
  """
  acc += history_fine.history['accuracy']
  val_acc += history_fine.history['val_accuracy']
  loss += history_fine.history['loss']
  val_loss += history_fine.history['val_loss']
  plt.ylim([0.8, 1]) # set limit
  plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
  """

  loss, accuracy = model.evaluate(test_dataset)
  print('Test accuracy :', accuracy)
  loss, accuracy = model.evaluate(test_dataset_batch)
  print('Test_batch accuracy :', accuracy)

  pabou.see_training_result(total_history, accuracy, 'train_result_finetuned.png', single_output = True, fig_title ='fine tune' )

  (tf_dir, h5_file, h5_size) = pabou.save_full_model(app , model, os.path.join('checkpoint', app), signature='False')
  print('model saved. %s, %s, %0.2f mb' %(tf_dir, h5_file, h5_size))
 
  ##############################
  # create quantization aware model
  # used by TFlite convert
  ##############################
  import tensorflow_model_optimization as tfmot
  quantize_model = tfmot.quantization.keras.quantize_model

  q_aware_model = None
  try:
    q_aware_model = quantize_model(model)
    q_aware_model.compile(loss=loss,
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
    metrics=['accuracy'])

    history_q_aware = q_aware_model.fit(train_dataset_batch, epochs=1, 
    validation_data=validation_dataset_batch , verbose = 1)

    print('created quantization aware model')

    loss, accuracy = q_aware_model.evaluate(test_dataset)
    print('Test accuracy on quantization aware model:', accuracy)

  except Exception as e:
    print('cannot create quantization aware model: ', str(e))
    # ValueError: Quantizing a tf.keras Model inside another tf.keras Model is not supported.
  

  ######################################################
  # DO some predictions on KERAS model. 
  # WARNING. only ONE output head for now
  ######################################################

  n=10
  # retrieve thruth labels

  #_,labels = iter(test_dataset.take(n).as_numpy_iterator().next()) # do not work, array has only ONE element

  # 10 labels from dataset with element = 1 pic
  x = test_dataset.batch(n).take(1) 
  x = x.as_numpy_iterator()
  _ , l = x.next()
  labels_batch = np.squeeze(l,axis=1) #array([2, 0, 0, 2, 1, 2, 2, 1, 1, 2])

  # 32 labels from dataset with element = batch size pic

  # returns numpy
  x = test_dataset_batch.take(1)
  x = x.as_numpy_iterator()
  _ , labels_batch_1 = x.next() # array([1, 2, 1, 2, 0, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2, 1, 2, .... 32
  
  # returns tensors
  for _, labels_batch_2 in test_dataset_batch:
    break # get 1rs one
  #<tf.Tensor: shape=(32,), dtype=int32, numpy= array([2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 2, 1, 0, 0, 2, 1, 0, 1, 2, 2, 1, 1,

  # returns tensors
  for _, labels_batch_3 in test_dataset_batch.take(1):
    pass # should only run loop once

  # .all() true is all array elemnts are true. .any() is any
 
  # not equal. differents calls to dataset seems to return differents input ??
  #assert np.array_equal(labels_batch_1, labels_batch_2.numpy())
  #assert np.array_equal(labels_batch_1, labels_batch_3.numpy())
  #assert np.array_equal(labels_batch_2.numpy(), labels_batch_3.numpy())
  #assert np.array_equal(labels_batch, labels_batch_1[:nb])

  #print(labels_batch)
  #print(labels_batch_1[:n])
  #print(labels_batch_2.numpy()[:n])
  #print(labels_batch_3.numpy()[:n])

  
  # predict using dataset. no sigmoid if multiclass (softmax altready there)
  p = model.predict(test_dataset.take(n))
  #p = tf.nn.sigmoid(p) # convert to 0 1. NOTE. sum is not 1. this is not a softmax NOTE: why not the argmax before sigmoid. maybe this is to use tf.where later in case of binary classification
  
  print('predictions with keras model, on dataset: type %s len %d shape %s ' %(type(p), len(p), p.shape))

  # predict using array of images
  x = test_dataset.batch(n).take(1)
  x = x.as_numpy_iterator()
  x,_ = x.next()
  image_batch = np.squeeze(x,axis=1)

  #predictions = model.predict_on_batch(image_batch)  # (32, 1)   already a batch. could run predict on a batch only of a larger dataset
  predictions = model.predict(image_batch)  # (16, 3)  16 array of 3 float32. NOT summing to zero as logits=True

  # for binary 16 array of 1 floats . squeeze convert into one array of 16 floats
  # predictions = predictions.squeeze(axis=-1)  # extra dim. need to squeeze [ 4.2298803]], dtype=float32)
  #array([-7.122643  , -4.3386745 , -6.318095  ,  2.5513978 ,  5.8884425 ,

  print('predictions on image batch %s: type %s len %d shape %s' %(image_batch.shape, type(predictions), len(predictions), predictions.shape))
  #predictions = tf.nn.sigmoid(predictions) # not needed if softmax

  # NOT equal. maybe because input is not the same
  try:
    assert np.array_equal(predictions,p)
  except Exception as e:
    print("predictions from test dataset and test dataset batch do not match ")

  # get labels and images simultaneously

  # try both test and test_batch

  for images_batch , labels_batch in test_dataset_batch: # batch dataset
    #print('predict with model.predict()')
    #predictions = model.predict(images_batch)

    print('predict with model()')
    predictions = model(images_batch) # faster than model.predict
    break # get only one batch 

  # prediction[i] is an array of nclass floats

  for i in range(n):
    print('%d %d' %(np.argmax(predictions[i]), labels_batch[i].numpy() ), end = ' ') # should be the same

  ###############################################
  # plot actual prediction
  # will use labels strings , class names
  ##############################################
  if not arg['headless']:

    ds = test_dataset.take(9)
    #ds = test_dataset_batch.take(1) # plot image expect dataset with element = 1 pic, as we iterate on it

    ### WARNING
    # do not generate predictions now and then mess around with data set (iterate on it); labels and predictions will not be on the same for loop in dataset and will be out of synch
    #predictions = model.predict(ds) # .predict accept dataset  cannot call model(ds) 
    """
    # model.predict()
    A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
    A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
    A tf.data dataset.
    A generator or keras.utils.Sequence instance. 
    """
  
    pabou.plot_9_images_1 (plt, ds, (6,6), 'actual predictions', class_list = class_names, do_predictions = True, model = model)
   

  # test set with batch size = 1
  for  single_image, single_label in test_dataset.take(2*n): # returns x times a single image TensorShape([1, 160, 160, 3]) TensorShape([1])
      predictions = model.predict(single_image)
      if np.argmax(predictions[0]) != single_label.numpy(): # for predict on one sample, prediction if array of one array
        print(np.argmax(predictions[0]),single_label)
      else:
        print('OK', end = ' ')
  print(' ')

  """
  # where for binary classification
  # Returns the indices of non-zero elements, or multiplexes x and y
  # condition, x=None, y=None, name=None
  # round to 0 or 1
  predictions = tf.where(predictions < 0.5, 0, 1) # The result is taken from x where condition is non-zero or y where condition is zero.
  print('Predictions:\n', predictions.numpy())
  print('Labels:\n', label_batch)
  """


  

  #########################################################
  # create label file
  # label file name is included in for TFlite metadata. 
  # also to be used in inferences, ie lili_run
  # use 'maixhub' format, ie exec()    labels = ['', '', '' ] 
  # or standard tf format , one label per line
  # order is paramount
  ##########################################################
  print('create label file')

  # maixhub format
  with open(label_file + 'M', 'w') as fp:
    fp.write('labels = [')
    for i, label in enumerate(class_names): # alphabetical order by default
      #fp.write(label + ' ')
      fp.write("'")
      fp.write(label)
      fp.write("'")

      # last label
      if i == len(class_names) - 1:
        fp.write(']')
      else:
        fp.write(',')

  with open(label_file + 'M', 'r') as fp:
    print(fp.read())
  
  # TF format
  with open(label_file, 'w') as fp:
    for label in class_names:
      fp.write(label); fp.write('\n')
  
  with open(label_file, 'r') as fp:
    print(fp.read())


  ######################################################################
  # TFLITE
  ######################################################################
  print('convert to Tflite')

  output_range =["0","1"]
  input_range = ["0", "255"]
  meta = {
  "name":"candyspencer",
  "description":"vont en bateau\n",
  "version":"v2",
  "author":"pabou. pboudalier@gmail.com. Meaudre Robotics",
  "license":"GNU GPL 3",

  "input_1_name":"input image",
  "input_1_description": "(160,160,3) image",
  "input_1_dimension_name": "(160,160,3)",
  "input_1_min":input_range[0],
  "input_1_max":input_range[1],
  "input_1_associated_files": "None",

  "output_1_name":"softmax output",
  "output_1_description":"classes prediction",
  "output_1_dimension_name": "(3,) float32",
  "output_1_min":output_range[0],
  "output_1_max":output_range[1],
  "output_1_associated_files": "None",

  "label_description":"list of string labels. use python's exec() to create labels variable",
  }

  # see also content https://github.com/tensorflow/tflite-support/blob/4cd0551658b6e26030e0ba7fc4d3127152e0d4ae/tensorflow_lite_support/metadata/metadata_schema.fbs#L640

  print('create dataset to be used as representative dataset generator for TFlite conversion')
  # could create array and have the gen do:  for x in array[:100]   yield(x)

  # or use a specific dataset . CARE: gen returns img, label
  representative_set = train_dataset.take(200)
  # expect gen to do: for x in this dataset, ie will retreive 100 image one by one and yield them. because batch = 1

  # Cannot batch tensors with different shapes in component 0. First element had shape [16,160,160,3] and element 1 had shape [10,160,160,3]. [Op:IteratorGetNext]
  # so use drop = true
  # representative_set = train_dataset.batch(10, drop_remainder = True) 
  
  print('representative set', type(representative_set), len(list(representative_set)))

  pabou.save_all_tflite_models(app, representative_set, meta, model, q_aware_model)

  return(model, h5_size)


##################################################
# model maker
##################################################
def lite_make_model_classification(input_dir, tflite_file_name):
  from tflite_model_maker import model_spec
  from tflite_model_maker import image_classifier
  from tflite_model_maker.config import ExportFormat
  from tflite_model_maker.config import QuantizationConfig
  from tflite_model_maker.image_classifier import DataLoader
 
  data_loader = DataLoader.from_folder(input_dir, shuffle = True) # sub dir are labels. returns dataloader (dataset? )
  # Image analysis for image classification load images with labels. 
  # INFO:tensorflow:Load image with size: 286, num_label: 3, labels: background, lili, pabou.
  
  print('model maker loaded %d pictures ' % len(data_loader))

  train_data, tmp = data_loader.split(0.8) 
  test_data, validation_data = tmp.split(0.5)

  # Gets model spec by name or instance, and init with args and kwarges.
  # base model for classification
  spec = model_spec.get('mobilenet_v2')
  # https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/model_spec    all available models

  # efficientnet_lite0  13mb file Trainable params: 3,843 Non-trainable params: 3,413,024
  # mobilenetv2  9mb file Trainable params: 3,84 Non-trainable params: 2,257,984

  # WARNING: model has fixed input size. dataset must match
  model = image_classifier.create(train_data, validation_data=validation_data, model_spec=spec, \
    shuffle = True, use_augmentation = False, epochs = 20, do_train = True 
    )  # The default EfficientNet-Lite0. 
  

  model.summary()
  loss, accuracy = model.evaluate(test_data)

  quant_fp16 = QuantizationConfig.for_float16()

  quant_dynamic =  QuantizationConfig.for_dynamic()

  #quantization_steps Number of post-training quantization calibration steps to run.

  quant_tpu =  QuantizationConfig.for_int8(
    data_loader,
    inference_input_type=tf.uint8,
    inference_output_type=tf.uint8,
    supported_ops=tf.lite.OpsSet.TFLITE_BUILTINS_INT8
)

  model.export(export_dir='.', tflite_filename=tflite_file_name, quantization_config=None)



##################################################
##################################################
# main
##################################################
##################################################

model = None

classes_dir = get_input_dir()

  # build datasets
(train_dataset, validation_dataset, test_dataset, class_names, train_dataset_batch, validation_dataset_batch, test_dataset_batch) \
  = get_dataset(classes_dir)

model = None

# load model
if arg['process'] == 'load':
  print('load model')
  (model, h5_size,_) = pabou.load_model(app,'tf' )
  if model == None:
    print('cannot load model')
    sys.exit(1)
  else:
    print("Model loaded. h5_size %d\n input %s\n output %s" %(h5_size, model.input, model.output))
    acc_list, confusion_matrix = pabou.model_evaluate(model, train_dataset_batch, None, func=func)
    
# fit model
if arg['process'] == 'transfert': 
  print('transfert learning from SavedModel. train and save model, FULL and Lite')
  # use both dataset for pic plot and dataset_batch for training

  (model, h5_size) = transfert_learning(train_dataset, validation_dataset, test_dataset, class_names, train_dataset_batch, validation_dataset_batch, test_dataset_batch)

  # evaluate from dataset. ONLY works for keras model
  # if x dataset with labels, no need to specify y
  # handle multi outputs
  # returns list of ALL accuracy and confusion matrix. 
  if model != None:
    acc_list, confusion_matrix = pabou.model_evaluate(model, train_dataset_batch, None, func=func)

  else:
    print('transfert learning failed')
    sys.exit(1)

if arg['benchmark'] and model != None:

  nb = 2 # number of batch  (batch is 32 or 1)

  # benchmarks full model
  # on nb batches of 32/1. anyway will get what is in the dataset
  # benchmark do not need labels string. will only compare int
  # expects batched dataset. 
  # will do nb*batch size single inference or nb batched inferences

  pabou.bench_full(name_of_model, model, test_dataset_batch, h5_size, func=func, nb=nb)

  pt = PrettyTable()
  pt.field_names = pabou.bench_field_names

  # benchmarks all TFlite models
  for file in pabou.all_tflite_files:
    file = os.path.join(pabou.models_dir, app + file)
    metadata_json, label_list = pabou.get_lite_metadata(file) # None, None if no metadata
    print('list of labels from metadata : ', label_list)

    # use test_dataset_
    coral = ('edgetpu' in file)
    
    # nb is number of batches in test_dataset. so number of inference is nb x batch size
    try: 
      test_name = '' # set by bench_lite , unless there is an exception

      (test_name, nb_samples, elapse_ms, accuracy,  h5_size, TFlite_file_size, _) = \
      lite_inference.bench_lite_one_model('lili TFlite', file, test_dataset, h5_size, func=func, nb = nb, coral = coral, metadata = metadata_json )
      
      pt.add_row([test_name, nb_samples, elapse_ms, accuracy,  h5_size, TFlite_file_size, 0])

    except Exception as e:
      print('error TFlite benchmark for: %s' %file)
      pt.add_row(['error '+ file, 0 , 0, 0,  0, 0, 0])

  print(pt)


if arg['process'] == 'maker': 
  print('transfert learning from tflite, using model maker')

  tflite_file_name = 'model_maker.tflite'

  # train directly on images class (not dataset) . expects 224x244
  # same dir as for creating dataset
  lite_make_model_classification(classes_dir, tflite_file_name)

  # label list is list of lines or maixhub format
  metadata_json, label_list = pabou.get_lite_metadata(tflite_file_name) # None, None if no metadata
  print('list of labels from metadata : ', label_list)

  if arg['benchmark']:
    pt = PrettyTable()
    pt.field_names = pabou.bench_field_names
    # nb is number of batches in test_dataset

    # trained model maker expect 224x224
    for image, _ in test_dataset_batch:
      if image[0].shape[1] != 224:
        print('model maker expect images 224x244')
        sys.exit()
      
    # benchmark
    # pass a dataset which is compatible with model maker training run
    try:
      (test_name, nb_samples, elapse_ms, accuracy,  h5_size, TFlite_file_size, _) = \
      lite_inference.bench_lite_one_model('model_maker', tflite_file_name, test_dataset, 0 , func=func, nb = 5, coral = None, metadata = metadata_json )
      
      pt.add_row([test_name, nb_samples, elapse_ms, accuracy,  h5_size, TFlite_file_size, 0])
 
    except Exception as e:
      print('error model maker benchmark for: %s %s' %(tflite_file_name, str(e)))
      pt.add_row(['error model maker  ' + tflite_file_name, 0 , 0, 0,  0, 0, 0])

    print(pt)







plt.show()
print('end')

"""
print('look at conv filter')
# look at filters maps from internal conv layers
# first 8 layers output
layers_output = [layers.output for layers in model.layers[:8]]


# create model that input as usual and output above list of activation
# magic is that running this new model will use the model already trained  
a_inputs = model.layers[0].input # from trained model
a_outputs = layers_output
#################################
# CRASH
model_activation = tf.keras.Model(input = a_inputs, output = a_outputs) # multiple outputs

# get a single image
img = test_dataset.take(1).numpy()[0]
activations = model_activation.predict(img)
print('activations ', activations.shape, type(activations))
l=0 # nth layer to look at
print('activation for one layer', activations[l].shape)
nb_filers = activations[l].shape[4]

plt.figure(figsize=(10, 15)) # create current figure (reperes) = separate window

# show first 9 filters of layer l
for i in range(9): 
  ax = plt.subplot(3, 3, i + 1) # dans la figure precedemment cree. AUTO placement. create ONE sub
  plt.imshow(activations[i].numpy().astype("uint8"))
  # need to use .numpy. 'EagerTensor' object has no attribute 'astype'.
  # as original pixel is float 0 to 255, convert float to int. or rescale float to 0 1
  plt.title(str(i))
  plt.axis("off")

"""