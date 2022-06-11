#!/bin/bash

#sunnycase commented on 26 Oct 2021
#nncase doesn't support fp16，you should export float32 tflite without quantization.

echo 'compile kmodel from ' $1

home=/home/pi
cd /mnt/c/users/pboud/DEEP/transfert_learning/models

ls -lh $1.tflite

rm *.kmodel > /dev/null 2>&1


##### v5

#/home/pi/NNCASE-1.5/bin/ncc -v

#/home/pi/NNCASE-1.5/bin/ncc compile -i tflite -t k210 $1.tflite $1.kmodel \
#--dataset ../../data_augmentation/images/training/pabou --dataset-format image \
#--input-layout NHWC --input-type float32 --input-shape "1 3 160 160"


#cd /home/pi/ncc_01
#./ncc compile ../$1.tflite $1_01.kmodel -i tflite -o k210 \
#--dataset /mnt/c/users/pboud/DEEP/data_augmentation/images/training/pabou --dataset-format image 
#ls -lh $1_01.kmodel


# generates kmodel v4
#/home/pi/ncc_v02_beta4 infer $1.tflite $1.kmodel -i tflite  -o kmodel -t k210 --input-type float --inference-type float\

echo ' '
echo ' compile kmodel'
echo ' '

echo 'input type float'
/home/pi/ncc_v02_beta4 compile $1.tflite $1_float.kmodel -i tflite -o kmodel -t k210 --input-type float --inference-type float  --dataset-format image -v 
ls -lh $1_float.kmodel

# uint8 is default
# step4 quantize
echo 'input type uint8'
/home/pi/ncc_v02_beta4 compile $1.tflite $1_uint8.kmodel -i tflite -o kmodel -t k210 --input-type uint8 --inference-type uint8 \
--dataset ../../data_augmentation/datasets/pabou --dataset-format image 
ls -lh $1_uint8.kmodel



echo ' '
echo ' running inferences'
echo ' '

# one pic file in test. 
# generates *.bin in test directory. one pic =  12 bytes, ie 3 fp32 softmax

rm $home/*.bin
echo 'float'
$home/ncc_v02_beta4 infer $1_float.kmodel  $home/test   --dataset $home/test --dataset-format image 

echo 'uint8'
$home/ncc_v02_beta4 infer $1_uint8.kmodel  $home/test  --dataset $home/test --dataset-format image 




# If the input has 3 channels, ncc will convert images to RGB float tensors [0,1] in NCHW layout. 
#If the input has only 1 channel, ncc will grayscale your images. 
# Set to raw if your dataset is not image dataset for example, audio or matrices. 
#In this scenario you should convert your dataset to raw binaries which contains float tensors.

#-rwxrwxrwx 1 pi pi 8.5M Apr 14 17:04 lili_fp32.tflite
