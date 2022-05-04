all: dft

dft: DFT.cpp lodepng.cpp
	g++ -O3 -std=c++11 -o dft DFT.cpp lodepng.cpp

clean:
	rm ./dft

