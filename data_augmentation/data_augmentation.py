#!/usr/bin/env python
print('data augmentation using preprocessing image')

target_size=(224,224) # used by image generator

# to make sure env is activated, select interpreter (palette or status bar)

# 1: get pic. better is squared already
# 2: use windows powertoys to resize and square pic , to ./image_224_from_powertoys/<labels>
# 3:
# 4: will create ./datasets/<labels>/*.jpg   , augmented and original - for TF training
# 5: Maixhub OBJECT training expect dataset/images/<labels>/*.jpg and dataset/xml/<labels>/*.xml - can also omit <labels>
# 5: Maixhub CLASSIFICATION  dataset/<labels>/*.jpg 
# https://www.maixhub.com/ModelTrainingHelp_en.html


from posixpath import basename
from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
import tensorflow as tf
print('tf version:' , tf.__version__)

# 17 Fev 2022
# 
# https://inside-machinelearning.com/data-augmentation-ameliorer-rapidement-son-modele-de-deep-learning/
# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

import os,sys, platform
import shutil
import glob
# pip3 install Pillow
# anaconda install scipy

import cv2
print('open cv ', cv2.__version__)

import uuid

print('python executable: %s' %( sys.executable))
print('python version: %s' %(sys.version))
print('os: %s, sys.platform: %s, platform.machine: %s' % (os.name, sys.platform, platform.machine()))
print('system name: %s' %( platform.node()))
#print('path for module to be imported: %s\n' %(sys.path))
#print('path for executable: %s\n' % os.environ['PATH'])

# TUTO
#https://www.tensorflow.org/tutorials/load_data/images

# better to capture squared pic
input = 'images_224_from_powertoys' # resized and squared using powertoys

# input files are ACTUALLY in input224/lili/*.jpg since keras looks for class aka label sub-directory
current = os.getcwd()
input = os.path.join(current, input)

print('input files are in %s. should have been bulk rezised using powertoys' %(os.path.join(input, '<label>/*.jpg')))

pic_taken = len(glob.glob(input+'/*/*.jpg'))

#list_ = os.listdir(os.path.join(input, label)) 
print ('pic taken: %d in %s' %(pic_taken, input))

# maixhub compatible for Object detection datasets/images/ *.jpg   
# maixhub compatible for classification datasets/<labels>/ *.jpg   
  
output = os.path.join(current, 'datasets')
print('flow from dir: %s. output dir %s ' % (input,output))

labels = [os.path.basename(x) for x in glob.glob(input + '/*')] # class labels glob returns full path name

for label in labels:
    
    if not os.path.exists(os.path.join (output, label)):
        print('create %s' %(os.path.join (output, label)) )
        os.mkdir(os.path.join (output, label)) # to make sure create labels sub dir

    print('remove previous files for all labels')
    for f in os.listdir(os.path.join (output, label)): # remove all files
        os.remove(os.path.join(os.path.join (output, label), f))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define GENERATOR Image generator with process
# define ITERATOR gen.flow_from_dir. call .next(). returns tensors and can store to file 

"""
rotation_range pour faire pivoter une image de façon aléatoire sur une plage entre 0 et la valeur choisis (maximum 180 degré)
width_shift et height_shift sont des plages (en fraction de la largeur ou de la hauteur totale) à l’intérieur desquelles on peut redimensionner aléatoirement des images verticalement ou horizontalement.
shear_range est une plage permettant de rogner(découper) de manière aléatoire l’image
zoom_range permet de zoomer de manière aléatoire à l’intérieur des images
horizontal_flip retourne horizontalement des images de manière aléatoire (certaines seront retourner d’autres non)
fill_mode est la stratégie utilisée pour remplir les pixels nouvellement créés, qui peuvent apparaître après un pivotage, un rognage, etc
"""

# DOES NOT TOUCH INPUT SHAPE or DTYPE
# do not use validation split here. leave to downstream
# TF defaults is pixel 0 to 1  tf.keras.layers.Rescaling(1./255)    vs 0 to 255 

# rescaling to [0,1]
# can do 
#  here in generator with rescale
#  or preprecessing function that will be applied on each input. The function will run after the image is resized and augmented.

#  or when building dataset:
#  normalization_layer = tf.keras.layers.Rescaling(1./255) 
#  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#  image_batch, labels_batch = next(iter(normalized_ds))

# or include the layer inside your model definition to simplify deployment
# If you want to include the resizing logic in your model as well, you can use the tf.keras.layers.Resizing layer.

# HERE do not rescale in data augmentation. better to leave this logic closer to the model. pic are pic

# rotate 90 degre counterclock k time
# for some reasons, my input pic need this
def rot(img):
    return (tf.image.rot90(img, k=1))


# Generate batches of tensor image data with real-time data augmentation.
# horizontal flip: left to rigth ?  if true creates image upside down 
# vertical: upside down ?   seems opposite

augment_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    rescale = None, 
    preprocessing_function=rot ,
    brightness_range=[0.2,1.0],
    fill_mode='nearest'
    )

# validation_split = 0.2,
# IF use validation split. so later use: gen.flow from directory (subset = validation)

# better to use powertoys to bulk resize AND convert portrait to square. if original pic is portrait

# each call to .next() generates as many images as non augmented, returns tensors and and store them
# images under directory test (keras look for class subdir)
# classes list of classes or infered from dir structure
# class mode "categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels,

"""
training_dir = os.path.join(output, 'training')
validation_dir = os.path.join(output, 'validation')

# create dir structure  training/labels
if not os.path.exists(training_dir):
    os.mkdir(training_dir)

if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
"""


# if class = None flow_from_directory Found 62 images belonging to 2 classes.  (total = 76 and 80%)
# class = pabou Found 34 images belonging to 1 classes.

# DOCTRINE: whether create training validation directory here (and labels below) or leave that to downstream (eg create split dataset)

# if create train/val here, need to use class = and subset =
#      class_mode='binary', classes=[label], shuffle = True, subset = 'training',
#      if class = None, the augmented images goes in training, not training/label


# here do NOT split and create only labels sub dir

# class name: Only valid if "labels" is "inferred". This is the explicit list of class names (must match names of subdirectories). Used to control the order of the classes (otherwise alphanumerical order is used).
# validation_split	Optional float between 0 and 1, fraction of data to reserve for validation.
# subset	One of "training" or "validation". Only used if validation_split is set.

# pixels are float32.  labels type is from class_mode. 
#    if binary float32 tensor of 1s and 0s. if categorial, the labels float32 hot encoding. if int, the labels are an int32

# # .next() generates min(batch, nb pic) augmented and store to disk 
# if batch is larger, one .next() will generate as many new file on disk as there are on in the input

# if class = None, ALL augmented images goes in images, not images/label # Found 111 images belonging to 3 classes.
# class = ['pabou'] Found 50 images belonging to 1 classes. manage labels and output dir explicitly

for label in labels:

    print(label)
    augment_iter_disk = augment_gen.flow_from_directory(
        input, batch_size=500, target_size=target_size, class_mode='binary', classes=[label], shuffle = True, subset = None,
        save_prefix='A',save_format='jpg', save_to_dir=os.path.join(output, label), color_mode="rgb")  # save to dir and return tensors

        # this is an iterator: .next() to iterate / traverse
        # each .next() will return a tuple, but also store augmented files to disk


    # nb time original pic, ie go from nb to nb+1 pic
    nb = 3
    for i in range(nb):
        print('generate augmented')
        x = augment_iter_disk.next() # .next() generates min(batch, nb pic) augmented and store to disk 
        # returns tuple of ndarray

        # <class 'tuple'>
        # [ 74.,  74.,  75.]]]], dtype=float32)
        # array([0., 0., 0. 0.] dtype=float32)
        # len (x[0]) 62,  less than batch 200
        # x[0].shape (62, 224, 224, 3)
        # x[1].shape (62,)

    # using generator to copy original pic as well.
    # need rotation 90 degree for some image
    func = rot
    ImageDataGenerator(preprocessing_function=func).flow_from_directory(
    input, batch_size=500, target_size=(224,224), class_mode='binary', classes=[label], shuffle = True, subset = None,
    save_prefix='O',save_format='jpg', save_to_dir=os.path.join(output, label)).next()

    
pic = len(glob.glob(os.path.join (output, '*', '*.jpg')))
print('pic for training', pic) 
# original + n time original (if original less than batch)


"""
fn = os.listdir(os.path.join(input,label))
for f in fn:
    shutil.copy(os.path.join(input,label,f), output)
list_ = os.listdir(output) # dir is your directory path
print ('for training', len(list_))
"""

print('end')
assert (pic_taken * (nb+1) == pic)
sys.exit(0)
					 

###############################################################################
# tf dataset 

# can list(dataset) or list(dataset.as_numpy_iterator())
# .batch() Combines consecutive elements of this dataset into batches.

ds= tf.data.Dataset.range(10) # 10 batches of 1

list(ds) # returns tensors
list(ds.as_numpy_iterator()) # returns array

# get 1st element , ie 0
# iter(ds) convert dataset into python iterator, on which .next() can be done

iter(ds).next() 
iter(ds).next().numpy()
len(list(ds)) # 10

# for element in ds: or  in ds.take(3):
#     this is a tensor
#     convert to numpy with .numpy()

# for element in ds.as_numpy_iterator():
#     this is a numpy array



#for i in ds.take(1)   returns only one element, ie 0
#for i in ds:          returns 10 element

ds1 = ds.batch(10) # one new batch become 10 old batches , ie batch =1 element becomes batch = 10 elements 
iter(ds1).next().numpy()   # returns array of 10
list(ds1)
len(list(ds1)) # 1

# using tf.data
# .next() is only one batch.  dataset.take can returns multiple, additional: .batch, .repeat , ie continue if end of data set
# The Dataset.from_generator constructor converts the python generator to a fully functional tf.data.Dataset
# The constructor takes a callable as input, not an iterator ie def x: yield, not iter(x)
# the generator argument must be a callable object that returns an object that supports the iter() protocol (e.g. a generator function).
# output_types argument required
# output_shapes not required but is highly recommended as many TensorFlow operations do not support tensors with an unknown rank
# DEPRECATED: (output_shapes, output_types). Use output_signature instead
# The elements generated by generator must be compatible with either the given output_signature 

# only returns tensors. no save to disk
# KERAS python iterator as above
augment_iter_tensors = augment_gen.flow_from_directory(
    input, batch_size=100, target_size=(224,224), class_mode='binary', classes=[label],
    )

# iterators returns tuple of tensors . min of number of pic and batch
# if 119 pic and batch 100 .next 100 .next 19 .next 100 .next 19
for _ in range(4):
    (image, label) = augment_iter_tensors.next()
    print(type(image), len(image), image.dtype, image.shape)
    # <class 'numpy.ndarray'> 119 float32 (119, 224, 224, 3)
    print(type(label), len(label), label.dtype, label.shape)
    # <class 'numpy.ndarray'> 119 float32 (119,)


##### create tf dataset from keras (python) generator
# 1 element is 1 batch, ie take returns one batch of 100
ds = tf.data.Dataset.from_generator(
    lambda: augment_iter_tensors,
    #augment_gen, # `generator` must be a Python callable.
    args = None,
    output_signature = (tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32))
)

# if use shape=() , error. shape = None means unknows
# `generator` yielded an element of shape (100, 224, 224, 3) where an element of shape () was expected

print('dataset element spec: ' , ds.element_spec)
# dataset element spec:  (TensorSpec(shape=(), dtype=tf.float32, name=None), 
# TensorSpec(shape=(), dtype=tf.float32, name=None)

#ds.take(1) of type <class 'tensorflow.python.data.ops.dataset_ops.TakeDataset'>;
# take(1) returns one batch. take(2)  100, then 19
for (image,label) in ds.take(2):
    print(image.dtype, image.shape) #<dtype: 'float32'> (100, 224, 224, 3)
    print(label.dtype, label.shape)

# repeat 'duplicate' dataset

for (image, label) in ds.repeat(5):
    print(image.dtype, image.shape) 
    print(label.dtype, label.shape)

print('end')

