CC=nvcc
LINKER=nvcc
OBJS=blind_reg.o
CFLAGS=-g
LINKFLAGS=-lcublas -lgsl -lgslcblas

PROG=blind_reg

compile: $(OBJS)
	$(LINKER) -o $(PROG) $(LINKFLAGS) $(OBJS)

blind_reg.o: blind_reg.cu
	$(CC) $(CFLAGS) $(PREPROC) -c $<
