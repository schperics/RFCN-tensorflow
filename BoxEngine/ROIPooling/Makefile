PYTHON?=$(shell which python)
TF_INC := $(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
USE_OLD_EABI?=1
USE_GPU?=1
INCLUDES=-I/usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public
NVCC=/usr/local/cuda/bin/nvcc
ARCH=-arch=sm_30 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_61,code=compute_61

ifeq (${USE_OLD_EABI}, 1)
	EABIFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
else
	EABIFLAGS=
endif

ifeq (${USE_GPU}, 1)
	LINKDEPS=roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o
	GPUFLAGS=-DUSE_GPU=1
else
	GPUFLAGS=-DUSE_GPU=0
	LINKDEPS=roi_pooling.o roi_pooling_grad.o
endif

all: roi_pooling.so

roi_pooling_grad.o: roi_pooling_grad.cc
	g++ -std=c++11 ${EABIFLAGS} ${GPUFLAGS} -c -o roi_pooling_grad.o roi_pooling_grad.cc -I $(TF_INC) $(INCLUDES) -fPIC -O2

roi_pooling.o: roi_pooling.cc
	g++ -std=c++11 ${EABIFLAGS} ${GPUFLAGS} -c -o roi_pooling.o roi_pooling.cc -I $(TF_INC) $(INCLUDES) -fPIC -O2

roi_pooling.cu.o: roi_pooling.cu.cc
	$(NVCC) $(ARCH) -std=c++11 -c -o roi_pooling.cu.o roi_pooling.cu.cc -I $TF_INC $(INCLUDES) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

roi_pooling_grad.cu.o: roi_pooling_grad.cu.cc
	$(NVCC) $(ARCH) -std=c++11 -c -o roi_pooling_grad.cu.o roi_pooling_grad.cu.cc -I $TF_INC $(INCLUDES) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

roi_pooling.so: ${LINKDEPS}
	g++ -std=c++11 ${EABIFLAGS} -shared -o roi_pooling.so ${LINKDEPS} -I $(TF_INC) $(INCLUDES) -L$(TF_LIB) -ltensorflow_framework -fPIC -O2


#roi_pooling.so: roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o
#	g++ -std=c++11  -D_GLIBCXX_USE_CXX11_ABI=0 -shared -o roi_pooling.so roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o -I $(TF_INC) -fPIC -O2


clean:
	-rm *.so
	-rm *.o
