CC=g++
#CC=nvcc
LINKER=g++
#LINKER=nvcc
OBJS=blind_reg.o
CFLAGS=-g 
LINKFLAGS=
#LINKFLAGS=-lcublas -lgsl -lgslcblas

PROG=blind_reg

compile: $(OBJS)
	$(LINKER) -o $(PROG) $(LINKFLAGS) $(OBJS)

blind_reg.o: blind_reg.cpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<
