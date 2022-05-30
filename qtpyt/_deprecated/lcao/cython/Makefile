CFLAGS=$(python-config --cflags)
LDFLAGS=$(python-config --ldflags)
CC=gcc

#cython unit_tests.pyx # --> outputs fib.c

unit_tests: unit_tests.c
	$(CC) -o unit_tests unit_tests.c $(CFLAGS) $(LDFLAGS) 

#gcc -c unit_tests.c ${CFLAGS} # outputs fib.o
#gcc unit_tests.o -o unit_tests.so -shared ${LDFLAGS} # --> outputs fib.so
