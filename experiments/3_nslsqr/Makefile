CC=nvcc
HEADERS=-I ./
OBJS=aux.o aux2.o error.o kernel.o qmatrix.o quant.o nslsqr_mf.o nslsqr.o
INC=lmnslsqr
FLAGS=
LINK=-lcublas -lcurand -lm
EXE=exp3

# General object files
all: $(OBJS)
	$(CC) $(FLAGS) $(HEADERS) -c main.cu
	$(CC) $(LINKPATH) $(LINK) -o $(EXE) $(OBJS) main.o

quant.o: $(INC)/quant.h $(INC)/bintable.h $(INC)/quant.cu
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/quant.cu	

aux.o: $(INC)/aux.cu $(INC)/aux.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/aux.cu

aux2.o: $(INC)/aux2.cu $(INC)/aux2.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/aux2.cu	

qmatrix.o: $(INC)/qmatrix.cu $(INC)/quant.h $(INC)/qmatrix.h $(INC)/aux.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/qmatrix.cu	

kernel.o: $(INC)/kernel.cu $(INC)/kernel.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/kernel.cu

error.o: $(INC)/error.cu $(INC)/error.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/error.cu

nslsqr.o: $(INC)/nslsqr.cu $(INC)/nslsqr.h $(INC)/qmatrix.h $(INC)/aux.h $(INC)/kernel.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/nslsqr.cu

nslsqr_mf.o: $(INC)/nslsqr_mf.h $(INC)/nslsqr_mf.cu $(INC)/qmatrix.h $(INC)/aux.h $(INC)/kernel.h $(INC)/aux2.h
	$(CC) $(FLAGS) $(HEADERS) -c $(INC)/nslsqr_mf.cu

clean:
	rm *.o exe-*