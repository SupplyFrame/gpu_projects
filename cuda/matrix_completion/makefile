CC=nvcc
LINKER=nvcc
OBJS=gpu_impute.o
CFLAGS=-g
LINKFLAGS=-lcublas -lgsl -lgslcblas

PROG=gpu_impute

compile: $(OBJS)
	$(LINKER) -o $(PROG) $(LINKFLAGS) $(OBJS)

gpu_impute.o: gpu_impute.cu
	$(CC) $(CFLAGS) $(PREPROC) -c $<
