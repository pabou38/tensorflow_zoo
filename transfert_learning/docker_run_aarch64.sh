#!/bin/bash

# run container on PI4
# can only run from PI OS. cannot use this script from within container (ie with remote ssh and remote container)
# run app directly from within container

echo 'env variable overwrite arguments set in CMD in Dockerfile'
echo 'should run from dir: transfert_learning'
echo 'maps models directory from container into host file system'

echo $(pwd)

#--device		Add a host device to the container
#--privileged -v /dev/bus/usb:/dev/bus/usb 
#--device=/dev/video0:/dev/video0 \

#--privileged, Docker will enable access to all devices on the host 
#want to limit access to a specific device or devices you can use the --device flag.

docker run -it --rm --name lili_run -e app=lili -e model=li -e quant=fp32 \
-e DISPLAY=${DISPLAY} \
--privileged \
--device=/dev/video0:/dev/video0 \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-v $(pwd)/models:/home/pi/transfert_learning/models \
pabou/lili_aarch64:v1

# NOTE: run locally on PI4, or use VNC, or run headless. otherwize cannot connect to display 

# use below if docker file uses ENTRYPOINT 
#pabou/lili_aarch64:v1 -pload -t63 -mli -qfp32 -e

# use below if docker file uses only CMD
#pabou/lili_aarch64:v1 python lili_run.py -pload -t63 -mli -qfp32 -e

# argparse default
## ENTRYPOINT and CMD definition in Dockerfile
### -e  and CMD arguments at docker_run 
