all: assignment.cu
	nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -lcublas -lcurand -I common/inc -std=c++11 -o assignment.exe
