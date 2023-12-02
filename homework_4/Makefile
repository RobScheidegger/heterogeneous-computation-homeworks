all:
	nvcc src/benchmark.cu -o benchmark

test: all
	./benchmark

profile: all
	nsys profile -o benchmark_profile.nsys-rep ./benchmark 

clean:
	rm -f benchmark