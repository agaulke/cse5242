all:
	gcc -O3 -mavx2 -o db5242 db5242.c

clean:
	rm db5242.exe