###############################################################################
#
# Build script for project
#
###############################################################################

###############################################################################
# SOURCE VARS

BINDIR := .
EXECUTABLE	:= compressedIdx

CUFILES :=	compressedIdx.cu

INCLUDES := -I../../cudpp/include \
			# -I/Developer/GPU Computing/C/common/inc

USECUDPP := 1

# LIB += -L/usr/local/cuda/lib -lcudpp_emu

include ../../linux_build/common.mk



