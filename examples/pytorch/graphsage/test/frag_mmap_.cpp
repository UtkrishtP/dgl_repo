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
#include <string>
#include <unistd.h>

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
    auto first_write = 0;
    auto iterate_write = (int *)malloc(10 * sizeof(int));
    for (int i = 0; i < 10; i++)
        iterate_write[i] = 0;
    std::ifstream file("memory_sizes_1024.txt");
    std::vector<int> sizes;
    std::vector<int> sizes_;
    int size;
    int count = 0;
    // Read sizes from the file
    while (file >> size) {
        sizes.push_back(size);
        // if (size < 2*1024*1024) count++;
    }

    // std::cout << "Count = " << count << std::endl;
    // return 1;
    //  auto min_it = std::min_element(sizes.begin(), sizes.end());
    // auto max_it = std::max_element(sizes.begin(), sizes.end());
    // std :: cout << min_it << max_it << std::endl;
    // return 0;
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
    // system("sudo perf record -g -F 999 -o ./perf_region1.data & echo $! > ./perf_region1.pid");
    auto pid = getpid();
    if (!mad_) 
    { 
        // system((std::string("sudo strace -c -fp ") + std::to_string(pid) + std::string(" 2>&1 | tee ./strace_malloc_st.txt & echo $! > ./perf.pid")).c_str());
        // system((std::string("sudo perf record -g -F 999 -o ./perf_malloc_mt.data -p ") + std::to_string(pid) + std::string(" & echo $! > ./perf.pid")).c_str());
        // system(comm.c_str());
    }
    else {
        // system((std::string("sudo strace -c -fp ") + std::to_string(pid) + std::string(" 2>&1 | tee ./strace_mmap_st.txt & echo $! > ./perf.pid")).c_str());
        // system((std::string("sudo perf record -g -F 999 -o ./perf_mmap_off_mt.data -p ") + std::to_string(pid) + std::string(" & echo $! > ./perf.pid")).c_str());
        // system(comm.c_str());
        }
    for (int size : sizes) {
        auto start = std::chrono::system_clock::now();
        int* mmap_array;
        if (mad_) mmap_array  = (int*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        // if (mad_) madvise(mmap_array, size, MADV_HUGEPAGE);
        if (!mad_) mmap_array = (int *)malloc(size);
        // int *mmap_array_write = (int *)malloc(64 * sizeof(int));
        // size_array[i_++] = mmap_array;
        auto start1 = std::chrono::system_clock::now();
        time_taken_ += std::chrono::duration_cast<std::chrono::microseconds>(start1 - start).count();

        #pragma omp parallel num_threads(num_)
        {
            // auto start_time = omp_get_wtime();
            int tid = omp_get_thread_num();  // Get the thread ID
            int nthreads = omp_get_num_threads();  // Get total number of threads
            int chunk_size = (size / sizeof(int)) / nthreads;  // Base chunk size
            int start_index = tid * chunk_size;
            int end_index = (tid == nthreads - 1) ? (size / sizeof(int)) : start_index + chunk_size;
            for (int j = start_index; j < end_index; j++) {
                mmap_array[j] = 10;  // Each thread writes to its designated part
                // tmp += mmap_array[j];
            }
             
            // region1  += omp_get_wtime() - start_time;
        #pragma omp barrier  
        }
        first_write += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        for (int i = 0; i< 9; i++){
        auto t = std::chrono::system_clock::now();
        #pragma omp parallel num_threads(num_)
        {
            // auto start_time = omp_get_wtime();
            int tid = omp_get_thread_num();  // Get the thread ID
            int nthreads = omp_get_num_threads();  // Get total number of threads
            int chunk_size = (size / sizeof(int)) / nthreads;  // Base chunk size
            int start_index = tid * chunk_size;
            int end_index = (tid == nthreads - 1) ? (size / sizeof(int)) : start_index + chunk_size;
            for (int j = start_index; j < end_index; j++) {
                mmap_array[j] = 10;  // Each thread writes to its designated part
                // tmp += mmap_array[j];
            }
             
            // region1  += omp_get_wtime() - start_time;
        #pragma omp barrier  
        }
        iterate_write[i] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t).count();
        }

        // for (int j = 0; j < size / sizeof(int); j++) {
        //         mmap_array[j] = 10;  // Each thread writes to its designated part
        //     }
        // first_write += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        // auto t = std::chrono::system_clock::now();
        // for (int i = 0; i< 1; i++){
        // for (int j = 0; j < size / sizeof(int); j++) {
        //         mmap_array[j] = 10 + i;  // Each thread writes to its designated part
        //     }
        // }
        
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count();
        time_taken += end;
        
    }
    // system("sudo kill -SIGINT `cat ./perf.pid`");
    printf("Time taken mmap,%d %d %d %d\n", time_taken_, first_write, time_taken + time_taken_);
    for (int i = 0; i < 9; i++)
        printf("%d ", iterate_write[i]);
    printf("\n");
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