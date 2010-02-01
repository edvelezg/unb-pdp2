all: matMulVec matMulVecCPU
.PHONY: all clean

incrementArray:
	nvcc -O3 -o incrementArray incrementArray.cu
	./incrementArray
	rm incrementArray
	
matrixAdd:
	nvcc -O3 -o matrixAdd matrixAdd.cu
	./matrixAdd
	rm matrixAdd
	
matMulVec:
	nvcc -O2 -o matMulVec matMulVec.cu
	# ./matMulVec 1000
	# rm matMulVec

matMulVecCPU:
	g++ -g -o matMulVecCPU matMulVecCPU.c
	# ./matMulVecCPU 1000
	
clean:
	rm matMulVec matMulVecCPU