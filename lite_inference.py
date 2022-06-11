
import numpy as np
import os,sys,time, platform
import math
from prettytable import PrettyTable
import tensorflow as tf

# duplicate from pabou.py
# consider re integrate in pabou
bench_field_names = ["model", "samples", "inference (ms)",  "accuracy", "h5 size Mb", 'TFlite size Mb', 'params']

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


#########################################
# duplicate/extract from pabou.py
# cannot import both tensorflow and tflite_runtime.interpreter
# this module does NOT import tensorflow
# dedicated to benchmark and inference for Coral
# do NOT import tensorflow prior importing this module to use coral
#########################################

##########################################################
# single TF lite inference
#########################################################

# input list: list of input for multi head. batch dim WILL BE ADDED here
# interpreter already created and passed as parameter, so works for both normal tflite and edge TPU
# need for dict for multihead, mapping layer name to index
# add batch dim , cast to dtype, quantize 
# returns array (possibly multi output) of softmax array

def TFlite_single_inference(interpreter, input_list, d=None):
    # input_list[0] single sample (160,160,3); but COULD BE multi head. tensor or nparray
    # will be rescale to int8 if needed
    # input_list is a list of all inputs, to cover multi head.

    input_details= interpreter.get_input_details() # input_details is a list of dictionaries

     # for model 1 ist with one dict is input_details[0].
    """
    [{
        'name': 'serving_default_x:0', 
        'index': 0, 
        'shape': array([  1,  40, 483]), 
        'shape_signature': array([  1,  40, 483]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization':  (0.0, 0) ,
        'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0} ,
        'sparsity_parameters': {}
        }]
    
    INT8 only
     'dtype': <class 'numpy.uint8'>, 
     'quantization': (0.003921568859368563, 0), 
     'quantization_parameters': {'scales': array([0.00392157], ...e=float32), 'zero_points': array([0]), 'quantized_dimension': 0}, 
                         scales array([0.00392157], dtype=float32)
     'sparsity_parameters': {}}

    input_details[0]
    {'name': 'serving_default_velocity:0', 'index': 0, 'shape': array([ 1, 40]), 'shape_signature': array([-1, 40]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}

    input_details[1]
    {'name': 'serving_default_pitch:0', 'index': 1, 'shape': array([ 1, 40]), 'shape_signature': array([-1, 40]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}

    input_details[2]
    {'name': 'serving_default_duration:0', 'index': 2, 'shape': array([ 1, 40]), 'shape_signature': array([-1, 40]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}

    """

    output_details= interpreter.get_output_details() 

    """
    {
        'name': 'StatefulPartitionedCall:0', 
        'index': 48, 
        'shape': array([  1, 483]), 
        'shape_signature': array([  1, 483]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization': (0.0, 0), 
        'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 
        'sparsity_parameters': {}
    }

    INT8 only
    {
        'name': 'StatefulPartitionedCall:0', 
        'index': 60, 
        'shape': array([  1, 483]), 
        'shape_signature': array([  1, 483]), 
        'dtype': <class 'numpy.uint8'>, 
        'quantization': (0.00390625, 0), 
        'quantization_parameters': {'scales': array([0.00390625], ...e=float32), 'zero_points': array([0]), 'quantized_dimension': 0}, 
        'sparsity_parameters': {}
        }

    """

    # check if input is compatible with model definition, ie same number of heads
    assert (len(input_list) == len(input_details)) #input_list[0] is 1st or only input

    for i, detail in enumerate(input_details): # all heads
        
        # loop on multi head
        assert detail['dtype'] in [np.float32 , np.int32, np.int8, np.uint8]

        # Check if the input type is quantized, then rescale input data to uint8
        if detail['dtype'] in [np.int8, np.uint8]:
            # rescale input X_test: (100,) dtype('int32')
            input_scale, input_zero_point = detail["quantization"]
            input_list[i] = input_list[i] / input_scale + input_zero_point # turn to float  (100,)  dtype('float64')

        # do here or in caller. need to supply [1,w,h,ch] to invoque. cast to input_details to make sure
        input = np.expand_dims(input_list[i], axis = 0).astype(detail["dtype"])

        ####### manage multi head.  set model's input
        try:
            interpreter.set_tensor(detail['index'], input) # should run once per input tensor/head
        except Exception as e:
            print('LITE: exception when setting model input ', str(e) )
            return(None, -1)

    #print (d) # map name of output head to index used in input_details[index]
  
    # INFERENCE at least
    start = time.perf_counter()
    interpreter.invoke()
    inference_time_ms = (time.perf_counter() - start) * 1000.0 # in sec
    #print('lite invoke %0.1f ' %inference_time_ms)

    # output_details list of dict. same order as input
    # use get_tensor to retrieve softmax from output_details[0 always for single output] ['index']

    # can also use squeeze: Remove single-dimensional entries from the shape of an array. [[ ]] to []
    #output_data = inter.get_tensor(output_details[0]['index'])   # (1, 10)
    #softmax = np.squeeze(output_data) # from array([[1.36866065e-05, to array([1.36866065e-05, 

    if len(output_details) > 1: # multi output; TO DO

        predictions = [] # one prediction (,3) per output head. could be softmax or int8  in all case take argmax
        for out in output_details: # multi outputs
            pred = interpreter.get_tensor(out['index']) # array([[-124,   -6,    1]], dtype=int8) for TPU

            """
            if np.dtype(out['dtype']) in [np.int8, np.uint8]:
                scale, zero_point = out['quantization']
                pred = scale * (pred - zero_point)
            """

            predictions.append(pred) # output head

    else: # single output head. output_details[0] is a dict

        predictions = interpreter.get_tensor(output_details[0]['index']) 
        # already array of array as in model.predict 
        #[[]] array of softmax. covers multi output
        #array([[0.50812346, 0.07269488, 0.41918164]], dtype=float32)
        #array([[-124,   -6,    1]], dtype=int8) for TPU

    # Please note: TfLite fused Lstm kernel is stateful, so we need to reset the states.
    #interpreter.reset_all_variables() 

    return(predictions , inference_time_ms) # array of softmax. one per output head


###########################################################################
# benchmark TFlite for ONE quantized model, passed as parameter as file name
############################################################################

# call TFlite_single_inference
# only iteration (vs slide)
# look at ELAPSE and ACCURACY (vs simple inference). use test set as input
# display with pretty table
# params: any string, file name of TFlite model, number of inferences
# import either tf or Coral

########  on WINDOWS need runtime v13 , and compile with option -m13


# prediction on nb elements of dataset, ie nb x batch

def bench_lite_one_model(st, TFlite_file, dataset, h5_size, coral=False, nb = 10, func = None, metadata = None):

    # dataset used for benchmark
    print("\nLITE: running LITE benchmark: %s, for model: %s" %(st, TFlite_file))

    TFlite_file_size = round(float(os.path.getsize(TFlite_file)/(1024*1024)),1) # to print in table

    ##############################################################
    # create TFlite interpreter once for all. create from TFlite file
    # see also load_model(app, 'li', None, None)
    ##############################################################
    
    """
    To quickly start executing TensorFlow Lite models with Python, you can install just the TensorFlow Lite interpreter, instead of all TensorFlow packages. We call this simplified Python package tflite_runtime.
    The tflite_runtime package is a fraction the size of the full tensorflow package and includes the bare minimum code required to run inferences with TensorFlow Lite—primarily the Interpreter Python class. This small package is ideal when all you want to do is execute .tflite models and avoid wasting disk space with the large TensorFlow library.
    """

    if coral:
        try:
            delegate = tf.lite.experimental.load_delegate(EDGETPU_SHARED_LIB)
            interpreter = tf.lite.Interpreter(model_path = TFlite_file, experimental_delegates=[delegate]) # 2 bong on windows
            print('LITE: CORAL delegate loaded')
        except Exception as e:
            print('LITE: cannot load delegate %s with tf.lite: %s' %(EDGETPU_SHARED_LIB ,str(e)))
            #Linux: libedgetpu.so.1 macOS: libedgetpu.1.dylib Windows: edgetpu.dll
            #Now when you run a model that's compiled for the Edge TPU, TensorFlow Lite delegates the compiled portions of the graph to the Edge TPU.
            return(['FAILED edgetpu. load delegate', 0, 0, 0 ,  h5_size, TFlite_file_size ])
    else:
        interpreter = tf.lite.Interpreter(model_path = TFlite_file)
        print('LITE: does not use CORAL. ') 

    # do this once
    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print('LITE: cannot allocate tensors %s. CHECK run time version vs edgetpu compiler -m' %(str(e)))
        return(['FAILED edgetpu. tensor allocation', 0, 0, 0 ,  h5_size, TFlite_file_size ])
        # Failed to prepare for TPU. Failed precondition: Package requires runtime version (14), which is newer than this runtime version (13)

    input_details= interpreter.get_input_details() # input_details is a list of dictionaries
    print('LITE: input ', input_details[0]['shape'], input_details[0]['dtype'])
        
    if metadata != None:
        print('LITE: TFlite model name: %s. description: %s' %(metadata['name'] , metadata['description']))

    for a,_ in dataset.take(1):
        dataset_batch_size = a.shape[0]
        break
    print('LITE: benchmark dataset %d element of batch size %d'  %(len(dataset), dataset_batch_size))

    # WARNING: for MULTI OUTPUT, order in list of dict not then same as model definition 
    d={}    # mapping layer name as defined in model to index used in input, output details
    # d['picth] is index    input_details[index] ['index', 'shape' , etc ..]

    # TO DO redo for 'serving_default_input_2:0'
    for i, details in enumerate(input_details):
        name = details['name'] # 'serving_default_velocity_2:0'
        name = name.split('_') [-1] # 'velocity_2:0'
        name = name.split(':')[0] # 'velocity'
        d[name] = i # {'velocity': 0}

    error=0
    nb_inferences = 0
    elapse_ms = 0.0

    #######################################################
    # do all predictions 
    # prediction on nb elements of dataset, ie nb x batch

    for images,labels in dataset.take(nb):  # returns nb batch of eager tensors . each for returns [batch_size, w, h, ch]
        #assert dataset_batch_size == images.shape[0] # not true for last batch. use drop_remainder = True

        # for all images in one batch i: 0 to 32/or less
        for i in range(images.shape[0]): #  ie for all indexes 0 to batch size, ie all images in one batch of 32
        # for i , input in enumerate(images)

            # adding batch dim is done in inference
            #images = np.expand_dims(images[x], axis=0) # img.shape (1, 160, 160, 3)
            
            input = images[i]  # TensorShape([160, 160, 3]). taking indice remove batch dim
            # MAKE IT generic . for now single input
            input = input.numpy()

            input_array = [input] # the inference module expects an array, to cope with multi heads

            (prediction, inference_time_ms) = TFlite_single_inference(interpreter, input_array, d) # single sample
            #print(inference_time_ms)

            #if prediction == None: # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            if inference_time_ms == -1: #error
                return(['error ', nb_inferences, 0, 0 ,  h5_size, TFlite_file_size, None])

            elapse_ms = elapse_ms + inference_time_ms
            
            # make it generic, prediction is an array of softmax

            if func != None: 
                prediction = func(prediction[0]) # get argmax. 1st head

            if prediction != labels[i]:   # test if softmax agrees. NOTE: getting labels right is something else. need correct label metadata
                error = error + 1

            nb_inferences=nb_inferences+1

    #assert (count == nb*dataset_batch_size) # not true for last batch. use drop_remainder = True

    a= float(error) / nb_inferences
    a = 1 - a
    average_elapse_ms = elapse_ms / nb_inferences

    print("LITE: %s elapse %0.2f ms, %d samples. %d error, %0.2f accuracy" % (TFlite_file, average_elapse_ms, nb_inferences, error, a))

    # create table just for this test
    pt = PrettyTable()
    pt.field_names = bench_field_names
    pt.add_row([os.path.basename(TFlite_file), nb_inferences, round(average_elapse_ms,1), round(a,2) ,  h5_size, TFlite_file_size, 0 ])
    print(pt)
    print('\n')

    # return to print in main , as one pretty table row
    # 'models\\lili_TPU.tflite' models\lili_TPU_edgetpu.tflite 
    test_name = os.path.basename(TFlite_file)
    test_name = os.path.splitext(test_name) [0] # remove extension
    test_name = test_name.split('_')[1]
    return([test_name, nb_inferences, round(average_elapse_ms,1), round(a,2) ,  h5_size, TFlite_file_size, None])


