all: dft_mpi

dft_mpi: DFT.cpp lodepng.cpp
	mpic++ -O3 -std=c++11 -o dft DFT.cpp lodepng.cpp

dft: DFT.cpp lodepng.cpp
	g++ -O3 -std=c++11 -o dft DFT.cpp lodepng.cpp

clean:
	rm ./dft

