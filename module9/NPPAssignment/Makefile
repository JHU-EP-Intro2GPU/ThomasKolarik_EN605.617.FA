all: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -L /usr/local/cuda/samples/7_CUDALibraries/common/FreeImage/lib/linux/x86_64 -lcudart -lnppi_static -lnppc_static -lculibos -lfreeimage -I common/inc -I /usr/local/cuda/samples/7_CUDALibraries/common/UtilNPP -I /usr/local/cuda/samples/7_CUDALibraries/common/FreeImage/include -std=c++11 -o assignment.exe
