PRGS = main

CFLAGS  = -Wall -g -O3
CFLAGS += -std=gnu11
CFLAGS += -march=native -mavx2

LDLIBS=-lopenblas

.PHONY: all clean

all: $(PRGS)

clean:
	rm -vf *.o $(PRGS)
