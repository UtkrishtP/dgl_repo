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

int main(int argc, char **args) {
    // auto start = std::chrono::system_clock::now();
    // auto start1 = std::chrono::system_clock::now();
    auto time_taken = 0;
    auto time_taken_ = 0;
    auto create1 = 0;
    auto write1 = 0;
    std::ifstream file("memory_sizes_1024.txt");
    std::vector<unsigned long long int> sizes;
    int size;
    int count = 0;
    unsigned long long int total_size = 0;
    // Read sizes from the file
    while (file >> size) {
        total_size += size;
        // total_size += 256*1024;
        sizes.push_back(total_size);
        // if (size < 2*1024*1024) count++;
    }
    // for (int s : sizes){
    //     if (s < 0) printf("Error\n");
    // }
    // std::cout << "Count = " << count << std::endl;
    // return 1;
    //  auto min_it = std::min_element(sizes.begin(), sizes.end());
    // auto max_it = std::max_element(sizes.begin(), sizes.end());

    // sizes_.push_back(*min_it);
    // sizes_.push_back(*max_it);

    // int **size_array = (int **)malloc(sizes.size() * sizeof(int *));
    // int i_ = 0;
    // std::cout << "Enter size \n";
    // std::cin >> size;
    // size = atoi(args[1]);
    // size *= 1024;
    int num_ = atoi(args[1]);
    // std::cout << "With madivse\n";

    int mad_ = atoi(args[2]);
    // std::cin >> mad_;
    // for (int i = 0; i < 12969; i++) {
    int* mmap_array;
    // std::cout << total_size << std::endl;
    if (mad_) mmap_array  = (int*)mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    // if (mad_) madvise(mmap_array, total_size, MADV_HUGEPAGE);
    if (!mad_) mmap_array = (int *)malloc(total_size);
    auto start1 = std::chrono::system_clock::now();
    unsigned long long int start_index_, end_index_, size_;
    start_index_ = end_index_ = size_ = 0;
    // std::cin >> start_index_;
    for (auto size : sizes) {
    // for(int i = 0; i< 12969; i++){
    //     size = 256 * 1024;
        auto start = std::chrono::system_clock::now();
        // int shm_fd = shm_open("f", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        // auto create_ = std::chrono::system_clock::now();
        // create1 += std::chrono::duration_cast<std::chrono::microseconds>(create_ - start).count();
        // auto res = ftruncate(shm_fd, size) == -1;
        // auto res_ = std::chrono::system_clock::now();
        // write1 += std::chrono::duration_cast<std::chrono::microseconds>(res_ - create_).count();
        // int* mmap_array = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        // int *mmap_array_write = (int *)malloc(64 * sizeof(int));
        // size_array[i_++] = mmap_array;
        auto start1 = std::chrono::system_clock::now();
        time_taken_ += std::chrono::duration_cast<std::chrono::microseconds>(start1 - start).count();

        #pragma omp parallel num_threads(num_)
        {
            auto tid = omp_get_thread_num();
            auto nthreads = omp_get_num_threads();
            auto elements_per_thread = ((size - start_index_) / sizeof(int)) / nthreads;
            auto start_index = tid * elements_per_thread + start_index_;
            auto end_index = (tid + 1) * elements_per_thread + start_index_;

            // Clamp the end_index to the maximum allowed index
            end_index = (end_index > total_size / sizeof(int)) ? total_size / sizeof(int) : end_index;

            for (auto j = start_index; j < end_index; j++) {
                mmap_array[j] = 10;
            }
        }
        // std::cout << count++ << std::endl;
        start_index_ = size;
        // printf("Size = %lld\n", size);
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        time_taken += end;
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