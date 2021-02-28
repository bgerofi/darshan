## Example usage of Darshan for profiling Resnet-50 training on ImageNet using PyTorch on ABCI

Clone the modified fork safe and python compatible branch of Darshan:

```bash
$ cd ~/src/
$ git clone https://github.com/bgerofi/darshan.git
```

PyTorch's data loader **fork()**s dedicated I/O processes which although can see the MPI library can not make MPI calls. For this reason we are using Darshan in POSIX only (i.e., without MPI) mode. One must configure and compile it accordingly:

```bash
$ cd ~/src/darshan/darshan-runtime
$ ./configure CC=gcc --without-mpi --prefix=${HOME}/local/ --with-log-path=darshan-logs --with-jobid-env=NONE --with-log-path-by-env=DARSHAN_LOGPATH --with-mod-mem=64
$ make && make install
```

Configure and compile the Darshan utils as well:

```bash
$ cd ~/src/darshan/darshan-util/
$ ./configure CC=gcc  --prefix=${HOME}/local/
$ make && make install
```

Example job script running **one epoch** of pytorch_imagenet on 32 ABCI nodes using Darshan:

```bash
#!/bin/bash
#$ -l rt_F=32
#$ -l h_rt=1:00:00
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch+horovod/bin/activate

ILSVRC2012_DIR=/path/to/ILSVRC2012/

DARSHAN_LOG_DIR="${HOME}/darshan-logs/"
mkdir -p ${DARSHAN_LOG_DIR}

NUM_NODES=${NHOSTS}
NUM_PROCESSES_PER_NODE=4
NUM_PROCESSES=$(echo ${NUM_PROCESSES_PER_NODE} \* ${NUM_NODES}|bc)
BATCH_SIZE=64

mpiexec -n ${NUM_PROCESSES} -map-by ppr:${NUM_PROCESSES_PER_NODE}:node \
	-mca pml ob1 \
	-x LD_PRELOAD=${HOME}/local/lib/libdarshan.so \
	-x DARSHAN_ENABLE_NONMPI=1 \
	-x DARSHAN_MODMEM=9984 \
	-x DARSHAN_LOGPATH=${DARSHAN_LOG_DIR} \
	-x DXT_ENABLE_IO_TRACE=1 \
	-x DARSHAN_EXCLUDE_DIRS=${HOME}/venv/pytorch+horovod,/apps,/sys,/proc,/bb,/dev,/tmp,/etc \
	python3 ./pytorch_imagenet.py --epochs 1 \
		--train-dir ${ILSVRC2012_DIR}/train \
		--val-dir ${ILSVRC2012_DIR}/val \
		--batch-size ${BATCH_SIZE}
```

A few important points to note:

- `LD_PRELOAD` is set to preload the Darshan runtime library
- `DARSHAN_ENABLE_NONMPI` is set to 1 to disable MPI support
- `DARSHAN_MODMEM` is set to a large enough value to support logging all file operations, note that this area is mapped using an ANONYMOUS mmap() call and is faulted on demand, no preallocation of memory is needed
- `DXT_ENABLE_IO_TRACE` is set 1 to enable DXT mode (i.e., logging timestamps for each POSIX I/O call)
- `DARSHAN_EXCLUDE_DIRS` excludes system files, libraries, etc.

Once you run this script you should be getting a lot of files (one per individual process), something like these:

```bash
$ cd ${HOME}/darshan-logs/
$ ls *python*.darshan
acc12613_python3_id116835-116835_2-8-64864-15306073932494800613_1612775058.darshan
acc12613_python3_id116836-116836_2-8-64864-15306073932494800613_1612775058.darshan
acc12613_python3_id116837-116837_2-8-64864-15306073932494800613_1612775058.darshan
acc12613_python3_id116838-116838_2-8-64864-15306073932494800613_1612775058.darshan
acc12613_python3_id117654-117654_2-8-64890-15306073932494800613_1612775059.darshan
....
```
Darshan provides a tool to merge all these into a single large file:

```bash
$ cd ~/darshan-logs
$ ~/local/bin/darshan-merge --output resnet50-imagenet-nodes-32-PFS.darshan *python3_id*
```

You can then process the DXT log as you wish, for example you may verify that all files from ImageNet have been accessed:

```bash
$ ~/local/bin/darshan-dxt-parser resnet50-imagenet-nodes-32-PFS.darshan | grep JPEG | wc -l
1331264
```
