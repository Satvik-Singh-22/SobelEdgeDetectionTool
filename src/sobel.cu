#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        int gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
                 + input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

        int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                 + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

        int magnitude = min(255, (int)sqrtf((float)(gx * gx + gy * gy)));
        output[y * width + x] = (unsigned char)magnitude;
    }
}


bool readPGM(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string magic;
    file >> magic;
    if (magic != "P5") return false;

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(1);  // skip newline

    data.resize(width * height);
    file.read(reinterpret_cast<char*>(data.data()), data.size());

    return true;
}

bool writePGM(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());

    return true;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./sobel input.pgm output.pgm\n";
        return 1;
    }

    std::string inputPath = argv[1], outputPath = argv[2];
    std::vector<unsigned char> inputImage, outputImage;
    int width, height;

    if (!readPGM(inputPath, inputImage, width, height)) {
        std::cerr << "Error reading input image\n";
        return 1;
    }

    outputImage.resize(width * height);

    unsigned char *d_input, *d_output;
    size_t size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, inputImage.data(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobel_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    if (!writePGM(outputPath, outputImage, width, height)) {
        std::cerr << "Error writing output image\n";
        return 1;
    }

    std::cout << "Edge detection completed successfully.\n";
    return 0;
}