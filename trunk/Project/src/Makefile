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
CFLAGS=--device-emulation
LDFLAGS=-O2
SOURCES=PrefixSum.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=PrefixSum
OUTFILE=$(OUTDIR)/$(EXECUTABLE)

all: $(OUTFILE)
	
$(OUTFILE): $(OUTDIR) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo done

%.o : %.cu
	$(CC) -c $< $(CFLAGS) -o $@

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) $(OBJECTS)

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