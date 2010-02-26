#
#  Ed
#
#  Created by  on 2010-02-23.
#  Copyright (c) 2010 __MyCompanyName__. 
#  All rights reserved.
#

# If no configuration is specified, "Debug" will be used
ifndef CFG
CFG=Debug
endif

MKDIR=mkdir
#
# Configuration: Debug
#
ifeq "$(CFG)" "Debug"
OUTDIR=Debug
CC=nvcc
CFLAGS=-g -ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin" --device-emulation
LDFLAGS=-ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin" -O2

SOURCES=PrefixSum.cu

OBJECTS=$(SOURCES:.cu=.o)
	
EXECUTABLE=PrefixSum
EXECUTABLE2=Sample
OUTFILE=$(OUTDIR)/$(EXECUTABLE)

all: $(OUTFILE) $(EXECUTABLE2)

$(EXECUTABLE2): $(OBJECTS2)
		$(CC) $(OBJECTS2) $(LDFLAGS) -o $@
		@echo done
	
$(OUTFILE): $(OUTDIR) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo done

%.o : %.cu
	$(CC) -c $< $(CFLAGS) -o $@

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) $(EXECUTABLE2) $(OBJECTS)

endif

#
# Configuration: Release
#
ifeq "$(CFG)" "Release"
OUTDIR=Release
CC=nvcc
CFLAGS=
LDFLAGS=-O2
SOURCES=PrefixSum.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=PrefixSum
OUTFILE=$(OUTDIR)/$(EXECUTABLE)

all: $(OUTFILE)
	
$(OUTFILE): $(OUTDIR) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo done

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

# Pattern rules
$(OUTDIR)%.o : %.cu
	$(CC) -c $< $(CFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(OUTFILE) $(OBJECTS)

endif

#make

#mkdir -p "Debug"
#nvcc PrefixSum.o -ccbin "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin" -O2 -o Debug/PrefixSum
#cl : Command line warning D9024 : unrecognized source file type 'PrefixSum.o', o
#bject file assumed
#done