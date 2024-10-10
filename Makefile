
all: fortran 

fortran: src/fortran/src/dqrls.f
	mkdir -p target/fortran
	
	gfortran -c src/fortran/src/dqrls.f -o target/fortran/dqrls.o
	ar rcs target/fortran/libdqrls.a target/fortran/dqrls.o