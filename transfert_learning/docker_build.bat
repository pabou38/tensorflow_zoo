echo 'docker build on windows'

@rem runs on windows

set IMG="pabou/lili:v1"
echo 'target image: ' %IMG%

@rem need to run from transfert_learning directory, access local Dockerfile and use .. as build context
@rem set build context to .. in order to access all sources. cannot use ../ in host file system

echo current directory is  %cd%

docker build -t %IMG%  -f ./Dockerfile_lili ..

docker image ls
@rem 2.33GB


@rem history shows layer size
@rem docker image history %IMG%