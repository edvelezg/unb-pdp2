# SlickEdit generated file.  Do not edit this file except in designated areas.

# Make command to use for dependencies
MAKE=make
RM=rm
MKDIR=mkdir

# -----Begin user-editable area-----

# -----End user-editable area-----

# If no configuration is specified, "Debug" will be used
ifndef CFG
CFG=Debug
endif

#
# Configuration: Debug
#
ifeq "$(CFG)" "Debug"
OUTDIR=Debug
OUTFILE=$(OUTDIR)/CUDA.exe
CFG_INC=
CFG_LIB=
CFG_OBJ=
COMMON_OBJ=
OBJ=$(COMMON_OBJ) $(CFG_OBJ)
ALL_OBJ=

COMPILE=gcc -c    -g -o "$(OUTDIR)/$(*F).o" $(CFG_INC) "$<"
LINK=gcc  -g -o "$(OUTFILE)" $(OBJ) $(CFG_LIB)

# Build rules
all: $(OUTFILE)

$(OUTFILE): $(OUTDIR)  $(OBJ)
	$(LINK)

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

# Rebuild this project
rebuild: cleanall all

# Clean this project
clean:
	$(RM) -f $(OUTFILE)
	$(RM) -f $(OBJ)

# Clean this project and all dependencies
cleanall: clean
endif

#
# Configuration: Release
#
ifeq "$(CFG)" "Release"
OUTDIR=Release
OUTFILE=$(OUTDIR)/CUDA.exe
CFG_INC=
CFG_LIB=
CFG_OBJ=
COMMON_OBJ=
OBJ=$(COMMON_OBJ) $(CFG_OBJ)
ALL_OBJ=

COMPILE=gcc -c   -o "$(OUTDIR)/$(*F).o" $(CFG_INC) "$<"
LINK=gcc  -o "$(OUTFILE)" $(OBJ) $(CFG_LIB)

# Build rules
all: $(OUTFILE)

$(OUTFILE): $(OUTDIR)  $(OBJ)
	$(LINK)

$(OUTDIR):
	$(MKDIR) -p "$(OUTDIR)"

# Rebuild this project
rebuild: cleanall all

# Clean this project
clean:
	$(RM) -f $(OUTFILE)
	$(RM) -f $(OBJ)

# Clean this project and all dependencies
cleanall: clean
endif
