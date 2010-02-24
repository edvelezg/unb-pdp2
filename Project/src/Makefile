#
#  untitled
#
#  Created by  on 2010-02-23.
#  Copyright (c) 2010 __MyCompanyName__. 
#  All rights reserved.
#

CC=nvcc
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=PrefixSum.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=PrefixSum

all: $(EXECUTABLE)
	
$(EXECUTABLE): $(SOURCES)
	$(CC) $(LDFLAGS) $(SOURCES) -o $@

.cu.o:
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) $(OBJECTS)
