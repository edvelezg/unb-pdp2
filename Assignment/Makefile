all: matMulVec
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
	nvcc -O3 -o matMulVec matMulVec.cu
	./matMulVec 1000
	# rm matMulVec
	
clean:
	rm incrementArray matrixAdd