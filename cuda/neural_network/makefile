CC=g++
#CC=nvcc
LINKER=g++
#LINKER=nvcc
OBJS=nn.o
CFLAGS=-g 
LINKFLAGS=
#LINKFLAGS=-lcublas -lgsl -lgslcblas

PROG=nn

compile: $(OBJS)
	$(LINKER) -o $(PROG) $(LINKFLAGS) $(OBJS)

nn.o: nn.cpp nn.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<
