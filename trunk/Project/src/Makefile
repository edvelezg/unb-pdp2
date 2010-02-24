#
#  untitled
#
#  Created by  on 2010-02-23.
#  Copyright (c) 2010 __MyCompanyName__. 
#  All rights reserved.
#

CC=nvcc
CFLAGS=-c -deviceemu
LDFLAGS=-deviceemu -O2
SOURCES=PrefixSum.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=PrefixSum

all: $(EXECUTABLE)
	
# $(EXECUTABLE): $(SOURCES)
# 	$(CC) $(LDFLAGS) $(SOURCES) -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	
	# $(NVCC) -c %^ $(CFLAGS) -o $@
	# $(NVCC) -M %^ $(CFLAGS) > $@.dep
		
# .cu.o:
# 	$(CC) $(CFLAGS) $< -o $@

%.o : %.cu
	$(CC) -c $< $(CFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) $(OBJECTS)
