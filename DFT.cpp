#include <complex>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <mpi.h>

#include "lodepng.h"

#ifndef M_PI
constexpr auto  M_PI = 3.14159265358979323846;
#endif
#define rootRank 0

void loadPNG(std::string filename, unsigned& w, unsigned& h, std::vector<unsigned char>& image)
{
	unsigned char* imageData;
	unsigned err = lodepng_decode_file(&imageData, &w, &h, filename.c_str(), LCT_GREY, 8);
	if (err != 0)
	{
		std::cout << "Image decoder error: " << err << std::endl;
		exit(-1);
	}
	image.resize(w * h);
	memcpy(&image[0], imageData, w * h);
}

void savePNG(std::string filename, unsigned& w, unsigned& h, std::vector<unsigned char>& image)
{
	unsigned err = lodepng_encode_file(filename.c_str(), image.data(), w, h, LCT_GREY, 8);
	if (err != 0)
	{
		std::cout << "Image encoder error: " << err << std::endl;
		exit(-1);
	}
}

void RealToComplex(std::vector<unsigned char>& in, std::vector<std::complex<double>>& out)
{
	out.clear();
	out.reserve(in.size());
	for (auto r : in)
	{
		out.push_back(std::complex<double>(static_cast<double>(r), 0.0));
	}
}

void ComplexToReal(std::vector<std::complex<double>>&in, std::vector<unsigned char>& out)
{
	out.clear();
	out.reserve(in.size());
	for (auto c : in)
	{
		double length = sqrt(c.real() * c.real() + c.imag() * c.imag());
		out.push_back(length);
	}
}

void DFT(std::vector<std::complex<double>>& in, std::vector<std::complex<double>>& out, unsigned w, unsigned h, bool horizontal, bool inverse)
{
	out.clear();
	out.resize(in.size());

	#pragma omp parallel for
	for (unsigned i = 0; i < h; ++i)
	{
		for (unsigned k = 0; k < w; ++k)
		{
			std::complex<double> sum(0.0, 0.0);
			for (unsigned j = 0; j < w; ++j)
			{
				size_t addr = horizontal ? j + i * w : i + j * w;
				auto angle = (inverse ? -2.0f : 2.0) * M_PI * j * k / w;
				sum.real(sum.real() + in[addr].real() * cos(angle) - in[addr].imag() * sin(angle));
				sum.imag(sum.imag() + in[addr].real() * sin(angle) + in[addr].imag() * cos(angle));
			}

			if (!inverse)
			{
				sum *= 1.0 / w;
			}

			size_t addr = horizontal ? k + i * w : i + k * w;
			out[k + i * w] = sum;
		}
		// std::cout << i << std::endl;
	}
}

void DFT_partial(std::vector<std::complex<double>>& in, std::vector<std::complex<double>>& out, unsigned w, unsigned h, unsigned offset, unsigned lenCount, bool horizontal, bool inverse)
{
	out.clear();
	out.resize(in.size());

	#pragma omp parallel for
	// for (unsigned i = horizontal ? offset : 0; i < (horizontal ? offset+lenCount : h); ++i)
	for (unsigned i = offset; i < offset+lenCount; ++i)
	{
		for (unsigned k = 0 ; k < w ; ++k)
		{
			std::complex<double> sum(0.0, 0.0);
			for (unsigned j = 0; j < w; ++j)
			{
				size_t addr = horizontal ? j + i * w : i + j * w;
				auto angle = (inverse ? -2.0f : 2.0) * M_PI * j * k / w;
				sum.real(sum.real() + in[addr].real() * cos(angle) - in[addr].imag() * sin(angle));
				sum.imag(sum.imag() + in[addr].real() * sin(angle) + in[addr].imag() * cos(angle));
			}

			if (!inverse)
			{
				sum *= 1.0 / w;
			}

			size_t addr = horizontal ? k + i * w : i + k * w;
			out[k + i * w] = sum;
		}
		// std::cout << i << std::endl;
	}
}

void Filter(std::vector<std::complex<double>>& in, int radiusMin, int radiusMax, unsigned w, unsigned h)
{
	for (int x = radiusMin; x < radiusMax; ++x)
		for (int y = radiusMin; y < radiusMax; ++y)
			in[x + y * w] = std::complex<double>(0.0, 0.0);
}

void printcheck(std::vector<unsigned char>& buffer, size_t n){
	//todo
	for (size_t i = 0; i < n; i++)
	{
		printf(" %d",buffer[i]);
	}
	printf("\n");
}

void imageNormalization(std::vector<unsigned char>& buffer, std::vector<unsigned char>& out, size_t n){
	//todo
	out.clear();
	out.resize(buffer.size());
	for (size_t i = 0; i < n; i++)
	{
		out[i]=buffer[i]*255/buffer[0];
	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
    int size;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::string inName("bmei.png");
	std::string dftName("bmeiDFT.png");
	std::string dftFName("bmeiDFTF.png");
	std::string outName("bmeiOut.png");

	unsigned w = 0, h = 0;
	unsigned wtest = 3, htest = 3;
	std::vector<unsigned char> image;
	loadPNG(inName, w, h, image);

	std::vector<std::complex<double>> f1;
	std::vector<std::complex<double>> f2;
	std::vector<std::complex<double>> g_horizontal;
	std::vector<std::complex<double>> g_image;
	g_horizontal.clear();
	g_image.clear();
	g_horizontal.resize(w*h);
	g_image.resize(w*h);

	RealToComplex(image, f1);

	int div=w/size;
	int mod=w%size;
	std::vector<int> g_offset;
	std::vector<int> g_counts;
	for (size_t i = 0; i < mod; i++)
	{
		g_offset.push_back(i*(div+1)*w);
		g_counts.push_back((div+1)*w);
	}

	for (size_t i = mod; i < size; i++)
	{
		g_offset.push_back((mod*(div+1)+(i-mod)*div)*w);
		g_counts.push_back(div*w);
	}
	
	int l_offset, l_counts;
	if (my_rank<mod)
	{
		l_offset=my_rank*(div-1);
		l_counts=(div+1);
	}else
	{
		l_offset=(mod*(div+1)+(my_rank-mod)*div);
		l_counts=div;
	}

	if (my_rank==rootRank)
	{
		for (size_t i = 0; i < size; i++)
		{
			printf(" %d", g_offset[i]);
		}
		printf("\n");
		for (size_t i = 0; i < size; i++)
		{
			printf(" %d", g_counts[i]);
		}
		printf("\n");
		fflush(stdout);
	}
	

	// DFT(f1, f2, w, h, true, false);
	DFT_partial(f1, f2, w, h,l_offset, l_counts, true, false);
	const void *l_send_pointer=&f2[l_offset*w];
	int l_send_counts=l_counts*w;

	if (my_rank==rootRank)
	{
		printf("DFT Done!");
		printf("\n");
		fflush(stdout);
	}

	MPI_Gatherv(l_send_pointer,l_send_counts,MPI_DOUBLE_COMPLEX,&g_horizontal[0],&g_counts[0],&g_offset[0],MPI_DOUBLE_COMPLEX,rootRank,MPI_COMM_WORLD);
	if (my_rank==rootRank)
	{
		DFT(f1, f2, w, h, true, false);
		for (size_t i = 0; i < w*h; i++)
		{
			if (f2[i]!=g_horizontal[i])
			{
				printf("Mismatch[%zu]\n",i);
				return 0;
			}
		}
		
		printf("Gather1 Done!");
		printf("\n");
		fflush(stdout);
	}
	MPI_Bcast(&g_horizontal[0],w*h,MPI_DOUBLE_COMPLEX,rootRank,MPI_COMM_WORLD);

	if (my_rank==rootRank)
	{
		printf("Broadcast Done!");
		printf("\n");
		fflush(stdout);
	}
	// DFT(f2, f1, w, h, false, false);
	DFT_partial(g_horizontal, f2, w, h,l_offset, l_counts, false, false);

	MPI_Gatherv(l_send_pointer,l_send_counts,MPI_DOUBLE_COMPLEX,&f1[0],&g_counts[0],&g_offset[0],MPI_DOUBLE_COMPLEX,rootRank,MPI_COMM_WORLD);

	if (my_rank==rootRank)
	{
		DFT(g_horizontal, f2, w, h, false, false);

		for (size_t i = 0; i < w*h; i++)
		{
			if (f1[i]!=f2[i])
			{
				printf("Mismatch[%zu]\n",i);
				return 0;
			}
		}

		std::cout << "DFT finished" << std::endl;
		ComplexToReal(g_horizontal, image);
		savePNG(dftName, w, h, image);

		// Frequency filtering
		int minFreq = 0;
		int maxFreq = 0;
		Filter(f1, minFreq, maxFreq, w, h);
		std::cout << "Filtering finished" << std::endl;
		ComplexToReal(f1, image);
		savePNG(dftFName, w, h, image);

		DFT(f1, f2, w, h, true, true);
		DFT(f2, f1, w, h, false, true);
		std::cout << "IDFT finished" << std::endl;
		ComplexToReal(f1, image);
		savePNG(outName, w, h, image);
	}

    MPI_Finalize();

	return 0;
}

