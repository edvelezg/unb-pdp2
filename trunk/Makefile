all: multMatbyVec
.PHONY: all clean

incrementArray:
	nvcc -O3 -o incrementArray incrementArray.cu
	./incrementArray
	rm incrementArray
	
matrixAdd:
	nvcc -O3 -o matrixAdd matrixAdd.cu
	./matrixAdd
	rm matrixAdd
	
multMatbyVec:
	nvcc -O3 -o multMatbyVec multMatbyVec.cu
	./multMatbyVec
	rm multMatbyVec
	
clean:
	rm incrementArray matrixAdd