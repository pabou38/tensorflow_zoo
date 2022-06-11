#!/usr/bin/env python3
# -*- coding: utf-8 -*-
			  
# do not use /usr/bin/python3. otherwize will not use env/bin/python3 and module import will fails
# If you have several versions of Python installed, /usr/bin/env will ensure the interpreter used is the first one on your environment's $PATH
# different behavior if ran with sudo


# pi@c74e12a4ad01:/workspaces/DEEP$ when running in VScode remote container
# CMD is not executed

import os, sys, time

print(os.getcwd()) 
# /workspaces/DEEP when running in VScode remote container

# see vscode setting.json "python.defaultInterpreterPath": "/home/pi/tf/bin/python3.9"
# !! python.pythonpath deprecated
print(sys.executable) 
# /bin/python3 when running in VScode remote container
 
#####
# when started in host OS, working dir is where it was started from. AND apps expects it be where the 'models' directory exist, ie transfert_learning
# when started from docker run , working dir is /home/pi/transfert_learning, as defined in docker file. this is a docker layer
# when started from remote container in VScode, working dir is /workspace/DEEP, as the entiere DEEP folder is opened in development container. this is a bind mount to host file system

try:
  # set in dockerfile
  if "indocker" in os.environ:
    if os.environ.get("indocker") == "yes": # docker run or VS code remote container


      # when in remote dev container, host file system is mounted in container, 
      #  and the cwd is one level up compared to what the app expects
      # ie ... DEEP vs DEEP/transfert_learning  (open remote container on DEEP to get access to all source files)

      # VS code remote container is started in /workspace/DEEP, but docker run in /home/pi/transfert_learning (layer)
      try:
        os.chdir("./transfert_learning") # for VS code
      except:
        pass

      run_from_docker = True
    else:
      run_from_docker = False

  else:
    run_from_docker = False

  print('running from within container ?: %s. %s' %(run_from_docker, os.getcwd()))

  # running from within container ?: True. /workspaces/DEEP/transfert_learning

except Exception as e:
  print('Exception environment', str(e))
  

import logging
import cv2
# USB webcam not supported in docker windows
# https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/hardware-devices-in-containers

# hack to support usb in wsl2
# https://devblogs.microsoft.com/commandline/connecting-usb-devices-to-wsl/
# https://www.youtube.com/watch?v=I2jOuLU4o8E

# on linux
# --device=/dev/video0:/dev/video0

import tensorflow as tf 
# import tf before cv2 generates error on Jetson: ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
import numpy as np

#pip install opencv-python  
# headless version has no support for GUI, ie imgshow

_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10

predict = True   # to benchmark camera

# omen  no predictions 30 fps
# 10 fps , tf
# 30 fps , tflite fp32, fp16
# 0 fps TPU
#  30 fps coral

# video capture
width = 640
height = 480

logging.getLogger("tensorflow").setLevel(logging.INFO)

print('transfert learning on Mobilenet')

print('tf ', tf.__version__)
print('cv ', cv2.__version__)

print('working dir: ', os.getcwd())

try:
  print('list physical GPU: ', tf.config.list_physical_devices('GPU'))
  print('GPU name: ', tf.test.gpu_device_name())
  print('built with CUDA ', tf.test.is_built_with_cuda())
except:
  pass


# pabou.py is one level up
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..') # this is in above working dir when getcwd is transfert_learning

# when running in a remote dev container with vscode remote container extension:
# dockerfile is under DEEP, to access both helpers.py and <app>/all.py
# using remote container extension: open folder DEEP, using docker file as the definition from dockerfile
#    connect VS frontend to a container with TF and all, getcwd /workspace/DEEP 
#      from the docker file, all python source is at the same level
#      when running DEEP/transfert_learning/lili_run.py 
#        the sys.path is ['/workspaces/DEEP/tra...t_learning', '..'
#        pabou.py is /workspace/DEEP , not in sys.path
#          CANNOT BE IMPORTED


#### NOT NEEDED with chdir above, but still good to understand

sys.path.insert(2, '.')

#print ('sys path used to import modules: %s\n' % sys.path)
try:
    print ('importing pabou helper')
    import pabou # various helpers module, one dir up
    import arg # arg parsing
    import lite_inference # single inference and benchmark
    #print('pabou path: ', pabou.__file__)
    #print(pabou.__dict__)
except:
    print('cannot import pabou. check if it runs standalone. syntax error there will fail the import')
    exit(1)


##################################################
# main
##################################################

try:
  arg = arg.parse_arg()
  print(arg)
except Exception as e:
  print('exception argument ', str(e))
  # when running with vscode remote container from DEEP, launch.json is in DEEP/.vscode

#usage: lili_run.py [-h] [-p {load,transfert,maker}] [-a APP] [-m {tf,li,h5}]
#                   [-q {fp32,default,adapt,quant-fp,TPU,GPU,18by8,TPU_edgetpu}] [-t THRESHOLD] [-e] [-b]

#{'process': 'load', 'app': 'lili', 'model': 'li', 'quant': 'GPU', 'threshold': 60, 'headless': False, 'benchmark': True}

app = arg['app']
model_type = arg['model'] # tf or lite
quant = arg['quant'] # for lite
threshold = arg['threshold']/100.0 # for float32 softmax , ie non quantized. arg is int
headless = arg['headless'] # boolean as defined in arg.py  default (ie no -e) is False , ie windowed, ie with screen

if run_from_docker:
  # overwrite SOME arg with ENV varianles, which can be passed at run time with docker run
  # env are str, whereas some arguments are defined as int or boolean
  if "app" in os.environ:
    app = os.environ.get("app")
  if "model_type" in os.environ:
    model_type = os.environ.get("model_type")
  if "quant" in os.environ:
    quant = os.environ.get("quant")
  if "threshold" in os.environ:
    threshold = os.environ.get("threshold")
    threshold = int(threshold) 
	
  if "headless" in os.environ:
    headless = os.environ.get("headless")  # -e headless=True
    # env is a string
    #windowed = windowed.lower == 'true'
    headless = str(headless) # convert str to boolean
  
# can use in print , because str(True) is 'True' and str(False) is 'False'.
print('app: %s model type: %s quant: %s headless: %s, threshold %0.1f' %(app, model_type, quant, headless, threshold))

windowed = not headless # backward compatibility. I used to use windowed


# load label from file
models_dir = pabou.models_dir # models_dir = 'models'
label_txt = pabou.label_txt
label_file = app + label_txt # name of label file 
label_file = os.path.join (models_dir, label_file) # label file path. to be created. expected by TFli[te metadata

# get labels from separate files, or TFlite metadata. use file as it works for Savedmodel as well
labels = []
with open(label_file) as fp: 
  #tmp = fp.read()
  #print('label file content: ', tmp)
  #exec(tmp) # Maixhub format # expect format to exec , ie labels = ['l1', '', '']
 
  lines = fp.readlines() # one label per line , strip leading and trailing newline
  labels = [x.strip('\n') for x in lines] 

print('labels from file: ', labels) # use labels[i]

tflite_file = app + '_'+ quant + pabou.lite_model_extension

if model_type in['tf', 'h5']:
  print('load keras model ', model_type)
  (model, model_size, _) = pabou.load_model(app, model_type ) # GTX960M compute capability = 5. jetson tegra 5.3
  if model == None:
    print('cannot load model for %s' %app)
    sys.exit(1)

  else:
    print("Model loaded. size %d Mb" %(model_size)) # model.input is a tensor
    # KerasTensor(type_spec=TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32, name='input_2'), name='input_2', description="created by layer 'input_2'")
    print('keras input layer: ', model.layers[0].name, model.layers[0].input.shape, model.layers[0].input.dtype)
    print('keras output layer: ', model.layers[-1].name, model.layers[-1].output.shape, model.layers[-1].output.dtype)
    input_size = (model.input.shape[1], model.input.shape[2])
    
if model_type == 'li':
  print('load tensorflow lite model %s' %tflite_file)
  # if edgetpu in name, will load coral
  (interpreter, model_size, (metadata_json, label_list)) = pabou.load_model(app,'li', lite_file = tflite_file )
  if interpreter == None:
    print('cannot load model for %s %s' %(app,tflite_file) )
    sys.exit(1)
  else:
    print("TFlite model %s loaded. size %d Mb" %(tflite_file, model_size))
    print('TFlite metadata ', metadata_json)
    print('label list from TFlite metadata', label_list)

    # input_details is a list of dict
    input_shape = interpreter.get_input_details()[0]['shape'] # array([  1, 160, 160,   3])
    print('input shape from TFlite model', input_shape)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input type from TFlite model', input_type)

    # test if model is quantized
    quantized = interpreter.get_input_details()[0]['dtype'] in [np.uint8, np.int8]
    print('quantized ? ' , quantized)
    if quantized:
      threshold = 0

    # to resize image from camera to model input
    input_size = (input_shape[1], input_shape[2])

    interpreter.allocate_tensors()

    # TO DO
    """
    process_units = metadata_json['subgraph_metadata'][0][ 'input_tensor_metadata'][0]['process_units']
    mean = 127.5
    std = 127.5
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    """

# Start capturing video input from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

counter, fps = 0, 0
last_idx = -1 # prediction from previous frame
successive_frame = 20 # number of frame with same prediction to trigger reco
frame = 0 # count successive frames with same prediction

i = 0
while not cap.isOpened():
  time.sleep(0.1)
  i = i + 1
  if i > 20:
    print('cannot open camera')
    sys.exit(1)


success, image = cap.read() # <class 'numpy.ndarray'>
print('camera ', image.shape, image.dtype) # (480, 640, 3) dtype('uint8')

start_time = time.time()

print('\n')

while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

    counter += 1
    
    # camera facing me
    image = cv2.flip(image, 1) # 0 means flipping around the x-axis and positive value (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes.
    
    # resize to model input
    input = cv2.resize(image, input_size)

    # webcam uint8
    if predict:
      if model_type == 'tf':
        input = np.expand_dims(input, axis=0).astype('float32')
        prediction = model(input)
      else:
        # TPU and edgetpu input uint8. webcamera is uint8
        if not quant in ['TPU', 'TPU_edgetpu']:
          input = input.astype('float32') # 
        else:
          input = input.astype('uint8') # should be already
          
        prediction, _ = lite_inference.TFlite_single_inference(interpreter, [input], d=None) # expect array of input (multi head) . batch dim added

      prediction = prediction[0]
      idx = np.argmax(prediction)

      # threshold 
      if prediction[idx] > threshold:
        text = labels[idx]
      else:
        text = 'unsure'
      
    else: # bench camera
      text = 'bench'
      prediction=[0,0,0]
      idx=0


    text_location = (_LEFT_MARGIN, 2  * _ROW_SIZE)
    cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (255,255,255), _FONT_THICKNESS)

    text_location = (_LEFT_MARGIN, 3  * _ROW_SIZE)
	# 255,255,255 is white
    cv2.putText(image, 'threshold %0.1f' %(threshold), text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (255,0,0), _FONT_THICKNESS)

    text_location = (_LEFT_MARGIN, 4  * _ROW_SIZE)
    cv2.putText(image, 'softmax %0.1f %0.1f %0.1f %0.1f'  %(prediction[0], prediction[1], prediction[2], prediction[3]), text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0,255,255), _FONT_THICKNESS)

    text_location = (_LEFT_MARGIN, 5  * _ROW_SIZE)
    cv2.putText(image, 'model %s: %s'  %(model_type, quant), text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0,255,255), _FONT_THICKNESS)

    # Calculate the FPS every n frames
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Show the FPS on image
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_LEFT_MARGIN, _ROW_SIZE) # top left
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,_FONT_SIZE, (255,0,0), _FONT_THICKNESS)

    # enough consecutive predictions to call it a detection ?
    if idx == last_idx: # predicted the same as last frame
      frame = frame + 1
      if frame > successive_frame: # enough successive frames with same prediction
        text_location = (_LEFT_MARGIN, 7  * _ROW_SIZE)
        cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

        if windowed == False:
          print('                       \r', end='' )
          print("%s %s\r" %(text,str(int(fps))), end ="" ) # with cr, without nl

    else:
      last_idx = idx
      frame = 0


    # Stop the program if the ESC key is pressed.
    try:
      if cv2.waitKey(1) == 27: # sleeps for X miliseconds, waiting for any key to be pressed. if a key is pressed, this function returns the ASCII code of key. Or returns -1 if no keys were pressed during that time.
        break
      
    except Exception as e:
      pass

	# this will fails in running from remote, ie ssh
    if windowed:
      try:
        cv2.imshow('image_classification', image) # show image
      except Exception as e:
        print('cannot show image. maybe running from remote ', str(e))
        sys.exit(1)
    
    # cv2.error: OpenCV(4.5.5) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support

    # is not build with GUI support, opencv-python-headless    4.5.5.64                 pypi_0    pypi
    # pip uninstall opencv-python-headless -y 
    #pip install opencv-python --upgrade
    
cap.release()
cv2.destroyAllWindows()


print('end')

    
