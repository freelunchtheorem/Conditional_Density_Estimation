# Nonparametric_Density_Estimation

## finished to-do's:
- change logging and data dumping to ml_logger --> done
- dumping individual single result pickles --> done
- running each task in a separate OS process --> done
- entropy regularization -> done
- data normalization --> done
- write ConfigRunner unittest to test entire I/O pipeline --> done
- generate nice plots each run and log them as well -> done
- fix GMM seed problem (GMM simulator is not reproducable) --> done
- two helpers.py existing (cde/density_simulation + cde/), merge into one --> done

## to-do's:
- setup docker (done) + launch script (not done)
- helpery.py, row 78, set n_jobs to 1 due to parallel error
- put sampling of datapoints back in run_single_task in order to avoid large memory footprint
- fix problems with tail risks est - sometimes takes extremely long
- fix GMM figure export problem

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

## Running Docker images
### run docker interactively
docker run -it <image> /bin/bash
    or
docker run -it --entrypoint /bin/bash <image>
    
### resume a container
docker exec -it <container-id> /bin/bash

# Modifying/Setting-up docker images
### kill all containers:
docker kill $(docker ps -q)

### commit changes to image
docker container ls
docker commit CONTAINER_ID tensorflow/tensorflow

### synchronize an image and upload it to docker hub
docker tag tensorflow/tensorflow ferreirafabio/nde:tf-cpu
docker push ferreirafabio/nde:tf-cpu

# bwUniCluster commands
see http://www.bwhpc-c5.de/wiki/index.php/Batch_Jobs for more info

### job shell script:

````
#!/bin/bash
#MSUB -l naccesspolicy=singlejob
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=48:00:00
#MSUB -l pmem=1000000mb
#MSUB -N config1
#MSUB -v PATH="$HOME/python3.6/bin:$PATH"
#MSUB -v PYTHONPATH="$HOME/python3.6:$PYTHONPATH"
#MSUB -v PYTHONPATH="/home/kit/fbv/gd5482/Nonparametric_Density_Estimation:$PYTHONPATH"

python /home/kit/fbv/gd5482/Nonparametric_Density_Estimation/cde/evaluation_runs/question1_noise_reg_x/configuration.py
````

### enqueue job in MOAB
```
msub -q fat job.sh
```
### check job status
```
showq -u $USER
```

### cancel job
```
canceljob <ID>
```

### use interactive mode for 'debugging'
```
msub  -I  -V  -l naccesspolicy=singlejob,pmem=64000mb -l walltime=0:24:00:00
```
for a fat node (32 CPUs) with 64GB RAM and then run python script manually

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

