#include <iostream>
#include <linux/mman.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <chrono>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
// const char* SHARED_MEMORY_NAME = "/dev/shm/my_la";
const size_t ARRAY_SIZE = 10;
// const size_t MEM_SIZE = 1<<10 * sizeof(int);
unsigned long long int MEM_SIZE = 400 * 1024; //294967296;

int main(int argc, char **args) {
    // auto start = std::chrono::system_clock::now();
    // auto start1 = std::chrono::system_clock::now();
    auto time_taken = 0;
    auto time_taken_ = 0;
    auto create = 0;
    auto pin = 0;
    auto write = 0;
    std::ifstream file("memory_sizes_1024.txt");
    std::vector<int> sizes;
    std::vector<int> sizes_;
    int size;
    int count = 0;
    int transfer = 0;
    // Read sizes from the file
    while (file >> size) {
        sizes.push_back(size);
        // if (size < 2*1024*1024) count++;
    }

    
    int num_ = atoi(args[1]);
    // std::cout << "With madivse\n";

    int mad_ = atoi(args[2]);
    // std::cin >> mad_;
    // for (int i = 0; i < 12969; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int size : sizes) {
        auto start = std::chrono::system_clock::now();
        int* mmap_array;
        // if (mad_) mmap_array  = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        // if (mad_) madvise(mmap_array, size, MADV_HUGEPAGE);
        mmap_array = (int *)malloc(size);
        // int *mmap_array_write = (int *)malloc(64 * sizeof(int));
        // size_array[i_++] = mmap_array;
        auto start1 = std::chrono::system_clock::now();
        time_taken_ += std::chrono::duration_cast<std::chrono::microseconds>(start1 - start).count();

        for (int j = 0; j < size/sizeof(int); j++) {
                mmap_array[j] = 10;  // Each thread writes to its designated part
                // tmp += mmap_array[j];
            }
        write += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        int* d_data;
        auto t = std::chrono::system_clock::now();
        cudaMalloc((void**)&d_data, size/sizeof(int));
        create += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t).count();
        t = std::chrono::system_clock::now();
        if (mad_) cudaHostRegister(mmap_array, size, cudaHostRegisterDefault);
        pin += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t).count();
        // Create CUDA stream
        
        
        // Asynchronous memory copy from host to device
        t = std::chrono::system_clock::now();
        if(num_) cudaMemcpyAsync(d_data, mmap_array, size/sizeof(int), cudaMemcpyHostToDevice, stream);
        else cudaMemcpy(d_data, mmap_array, size/sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemCpyAsync()
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t).count();
        transfer += end;
        end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        time_taken += end;
    }
    printf("Time taken mmap,%d %d %d %d %d %d\n", time_taken_, write, create, pin, transfer ,time_taken + time_taken_);
    return 0;

}