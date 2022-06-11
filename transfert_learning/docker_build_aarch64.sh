#!/bin/bash

echo 'docker build aarch64'

IMG="pabou/lili_aarch64:v1"

# run from transfert_learning.  use Dockerfile in transfert_learning 

pwd

# set build context to .. cannot use ../ in host file system
docker build -t $IMG  -f ./Dockerfile_lili_aarch64 ..

docker image ls

#history shows layer size
docker image history $IMG
