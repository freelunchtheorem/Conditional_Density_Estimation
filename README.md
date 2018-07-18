# Nonparametric_Density_Estimation

## to-do's:
- change logging and data dumping to ml_logger --> done
- dumping individual single result pickles --> done
- running each task in a separate OS process --> done
- entropy regularization -> done
- data normalization
- setup docker + launch script
- helpery.py, row 78, set n_jobs to 1 due to parallel error
- fix GMM seed problem (GMM simulator is not reproducable)
- generate nice plots each run and log them as well
- put sampling of datapints back in run_single_task in order to avoid large memory footprint
- fix problems with tail risks est - sometimes takes extremely long

# Citing
If you use NPDE in your research, you can cite it as follows:

```
@misc{npde2018,
    author = {Jonas Rothfuss, Fabio Ferreira},
    title = {Non-parametric Density Estimation},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/jonasrothfuss/Nonparametric_Density_Estimation}},
}
```


## tensorflow issues
- on workstations with ferreira account execute ```source activate p3.6```
- use tensorflow-gpu==1.2.0

### tf version 1.1
tensorflow version 1.1 works with installed cuDNN but "python3 density_estimator_tests.py" yields
"AttributeError: module 'tensorflow.contrib.distributions' has no attribute 'bijectors'", work-arounds on google don't help

### tf version > 1.2 <= 1.4 
importing tensorflow yields:
ImportError: /common/homes/students/ferreira/anaconda3/envs/p3.6/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so: undefined symbol: cudnnSetRNNDescriptor_v6
-> cuDNN 6 not properly installed, cuDNN 5 works
### tf version > 1.4
- cuda 9 and cudnn 7 required
- see https://www.tensorflow.org/install/install_sources#tested_source_configurations for cuDNN and cuda requirements

# Docker commands

# Setup stuff
## kill all containers:
docker kill $(docker ps -q)

## run docker interactively
docker run -it <image> /bin/bash oder
docker run -it --entrypoint /bin/bash <image>

## tag an image for uploading
docker tag tensorflow/tensorflow ferreirafabio/nde:tf-cpu
docker push ferreirafabio/nde:tf-cpu

## commit changes to image
docker container ls
docker commit CONTAINER_ID tensorflow/tensorflow

## resume a container
docker exec -it <container-id> /bin/bash

# CUDA/CudNN
```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
```

```
check libcudnn 
libcudnn.so.6 -> libcudnn.so.6.0.21 (changed)
libcudnn.so.5 -> libcudnn.so.6 (changed)
libcudnn.so.5 -> libcudnn.so.6
libcudnn.so.6 -> libcudnn.so.6.0.21
libcudnn is installed
```
