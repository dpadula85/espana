.PHONY: all clean 

all: inter.so

inter.so: inter.f90
	f2py  --opt='-O3 -funroll-loops' -lgomp --f90flags='-fopenmp' -c -m inter inter.f90 
	
clean:
	rm -rf inter.*.so
