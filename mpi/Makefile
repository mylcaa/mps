main:
	@mpic++ -o exe zad4_1.c
.PHONY: run
run:
	@mpiexec -n 4 ./exe vector.txt mat.txt
.PHONY: clean
clean:
	@rm -rf ./exe

