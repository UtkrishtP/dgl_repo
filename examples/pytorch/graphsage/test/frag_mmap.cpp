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

// const char* SHARED_MEMORY_NAME = "/dev/shm/my_la";
const size_t ARRAY_SIZE = 10;
// const size_t MEM_SIZE = 1<<10 * sizeof(int);
unsigned long long int MEM_SIZE = 400 * 1024; //294967296;

int main() {
    // auto start = std::chrono::system_clock::now();
    // auto start1 = std::chrono::system_clock::now();
    auto time_taken = 0;
    auto time_taken_ = 0;
    auto create1 = 0;
    auto write1 = 0;
    std::ifstream file("memory_sizes_1024.txt");
    std::vector<int> sizes;
    std::vector<int> sizes_;
    int size;

    // Read sizes from the file
    while (file >> size) {
        sizes.push_back(size);
    }

    //  auto min_it = std::min_element(sizes.begin(), sizes.end());
    // auto max_it = std::max_element(sizes.begin(), sizes.end());

    // sizes_.push_back(*min_it);
    // sizes_.push_back(*max_it);

    int **size_array = (int **)malloc(sizes.size() * sizeof(int *));
    int i_ = 0, mad_;
    std::cout << "madvise\n";
    std::cin >> mad_;
    for (int size : sizes) {
        auto start = std::chrono::system_clock::now();
        // int shm_fd = shm_open("f", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        // auto create_ = std::chrono::system_clock::now();
        // create1 += std::chrono::duration_cast<std::chrono::microseconds>(create_ - start).count();
        // auto res = ftruncate(shm_fd, size) == -1;
        // auto res_ = std::chrono::system_clock::now();
        // write1 += std::chrono::duration_cast<std::chrono::microseconds>(res_ - create_).count();
        // int* mmap_array = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        int* mmap_array = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (mad_) madvise(mmap_array, MEM_SIZE, MADV_HUGEPAGE);
        // int *mmap_array = (int *)malloc(size);
        // int *mmap_array_write = (int *)malloc(64 * sizeof(int));
        // size_array[i_++] = mmap_array;
        auto start1 = std::chrono::system_clock::now();
        time_taken_ += std::chrono::duration_cast<std::chrono::microseconds>(start1 - start).count();

        #pragma omp parallel num_threads(64)
        {
            int tid = omp_get_thread_num();  // Get the thread ID
            int nthreads = omp_get_num_threads();  // Get total number of threads
            int chunk_size = (size / sizeof(int)) / nthreads;  // Base chunk size
            int start_index = tid * chunk_size;
            int end_index = (tid == nthreads - 1) ? (size / sizeof(int)) : start_index + chunk_size;
            volatile int tmp;
            for (int j = start_index; j < end_index; j++) {
                mmap_array[j] = 10;  // Each thread writes to its designated part
                // tmp += mmap_array[j];
            }
            // mmap_array_write[tid] = tmp;
        }
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        time_taken += end;
        // for (int i = 0; i < size / sizeof(int); i++) {
        //     if(mmap_array[i] != 10){
        //         printf("Error\n");
        //         exit(1);
        //     }
        // }
    }
    printf("Time taken mmap,%d %d %d %d %d\n", create1, write1, time_taken_, time_taken , time_taken + time_taken_);
    return 0;

    // close(shm_fd);
    // munmap(mmap_array, MEM_SIZE);
    // shm_unlink(SHARED_MEMORY_NAME);

//     start = std::chrono::system_clock::now();
//     int* malloc_array = (int *)malloc(MEM_SIZE);
//     // memset(malloc_array, 10,  MEM_SIZE );
// // #pragma omp parallel for NTHREADS(64)
//     for (int i = 0; i < MEM_SIZE / sizeof(int); i++) {
//         malloc_array[i] = 10;
//     }
// // #pragma omp barrier
//     auto end = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start));
//     printf("Time taken malloc, %ld \n", end);
//     printf("malloc_array[0] = %d\n", malloc_array[0]);
//     return 0;
}