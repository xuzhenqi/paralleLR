INCLUDE=include
#CC=g++
CC=mpic++
CCFLAG=-std=c++11 -I$(INCLUDE) -DUSE_MPI -DDEBUG 

build/io.o: include/io.h src/io.cpp
	$(CC) -c $(CCFLAG) src/io.cpp -o build/io.o

build/demo.o: include/io.h include/model.h include/solver.h tools/demo.cpp
	$(CC) -c $(CCFLAG) tools/demo.cpp -o build/demo.o

bin/demo.bin: build/io.o build/demo.o
	$(CC) build/io.o build/demo.o -o bin/demo.bin -L/usr/lib/python2.7/config-x86_64-linux-gpu -lpython2.7

bin/test_reader.bin: test/test_reader.cpp build/io.o
	$(CC) $(CCFLAG) test/test_reader.cpp build/io.o -o bin/test_reader.bin -L/usr/lib/python2.7/config-x86_64-linux-gpu -lpython2.7

bin/test_model.bin: test/test_model.cpp include/model.h
	$(CC) $(CCFLAG) test/test_model.cpp include/model.h -o bin/test_model.bin 

all: bin/demo.bin bin/test_reader.bin bin/test_model.bin

clean: 
	rm bin/* build/*
