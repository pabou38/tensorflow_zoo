@rem 'docker run with bind mount models directory'
@rem 'make sure started from transfert_learning as it bind mount models from cwd'

echo %cd%

@rem env variable overwrite some arguments, when running from a container
@rem if env are not used, argument can be set at run time eg pabou/lili:v1 -e -t62. will overwrite CMD
@rem dockerfile's ENTRYPOINT and CMD overwrite argparse default 

@rem docker run -it --rm --name lili-run -e app=lili -e quant=fp32 -e threshold=62 pabou/lili:v1

docker run -it --rm --name lili-run -e app=lili -e quant=fp32 -e threshold=62 -v %cd%/models:/home/pi/transfert_learning/models pabou/lili:v1

@rem bug on windows. delete image
@rem docker: Error response from daemon: error while creating mount source path '/run/desktop/mnt/host/c/Users/pboud/Desktop/DEEP/transfert_learning/models': mkdir rem /run/desktop/mnt/host/c: file exists.

@rem [ WARN:0@7.493] global /io/opencv/modules/videoio/src/cap_v4l.cpp (902) open VIDEOIO(V4L2:/dev/video0): can't open camera by index
@rem will fail on windows. cannot access USB cam from within container

@rem to debug with VScode remote container, open DEEP folder. see DEEP/.devcontainer/devcontainer.json
@rem vscode will see all DEEP source tree, and will run execute in container defined in  "dockerFile": "../transfert_learning/Dockerfile_lili",

