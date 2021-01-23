
all:Gabriela 

CC=g++
RM=del /f
ALLFILES = config_cpu_access.h GabrielaGPUPADA.cpp GabrielaGPUPADA.h MatrixGPUPADA.h Optime_GPU.cl

LIB_DIR = ../lib


Gabriela: $(LIB_DIR)/gabriela32.dll $(LIB_DIR)/gabriela64.dll
	@echo compiled


$(LIB_DIR)/gabriela32.dll:gabriela32.o $(ALLFILES)
	$(CC) -m32 -shared -o $(LIB_DIR)/gabriela32.dll gabriela32.o
	$(RM) gabriela32.o

$(LIB_DIR)/gabriela64.dll:gabriela64.o $(ALLFILES)
	$(CC) -m64 -shared -o $(LIB_DIR)/gabriela64.dll gabriela64.o
	$(RM)  gabriela64.o

gabriela32.o:
	$(CC) -m32 -c GabrielaGPUPADA.cpp  -o  gabriela32.o

gabriela64.o:
	$(CC) -m64  -c GabrielaGPUPADA.cpp -o  gabriela64.o


