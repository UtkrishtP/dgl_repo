#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <chrono>
#include <omp.h>
#include <sys/wait.h>
const char* SHARED_MEMORY_NAME = "/dev/shm/my_la";
const size_t ARRAY_SIZE = 10;
// const size_t MEM_SIZE = 1<<10 * sizeof(int);
// unsigned long long int MEM_SIZE = 4294967296;
unsigned long long int MEM_SIZE = 4 * 1024;

int main() {
    auto start = std::chrono::system_clock::now();
    
    int shm_fd = 0;
    
    if ((shm_fd = open(SHARED_MEMORY_NAME, O_RDWR | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR)) == -1) {
        perror("shm_open");
        // delete[] array;
        return 1;
    }
    if (ftruncate(shm_fd, MEM_SIZE) == -1) {
        perror("ftruncate");
        // delete[] array;
        close(shm_fd);
        shm_unlink(SHARED_MEMORY_NAME);
        return 1;
    }
    int* mmap_array = (int*)mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED , shm_fd, 0);
    start = std::chrono::system_clock::now();
    // if (madvise(mmap_array, MEM_SIZE, MADV_HUGEPAGE) == -1) {
    //     perror("madvise");
    //     munmap(mmap_array, MEM_SIZE);
    //     // exit(EXIT_FAILURE);
    // }
    // memset(mmap_array, 10 , MEM_SIZE / sizeof(int));
// #pragma omp parallel for NTHREADS(64)
    
    auto end = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count());
    auto start1 = std::chrono::system_clock::now();
    for (int i = 0; i < MEM_SIZE / sizeof(int); i++) {
        mmap_array[i] = 10;
    }
    // int* shared_array = (int*)mmap(NULL, MEM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    // for (int i = 0; i < MEM_SIZE / sizeof(int); i++) {
    //     printf("shared_array[%d] = %d\n", i, shared_array[i]);
    // }
// #pragma omp barrier
    auto end1 = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start1).count());
    printf("Time taken mmap, %ld, %ld, %ld \n", end, end1, end + end1);
    printf("mmap_array[0] = %d\n", mmap_array[0]);

    // pid_t pid = fork();
    // if (pid == -1) {
    //     perror("fork");
    //     return 1;
    // }
    // if (pid == 0) {
    //     // Child process
    //     int* shared_array = (int*)mmap(NULL, MEM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    //     if (shared_array == MAP_FAILED) {
    //         perror("mmap");
    //         return 1;
    //     }
    //     for (int i = 0; i < MEM_SIZE / sizeof(int); i++) {
    //         printf("shared_array[%d] = %d\n", i, shared_array[i]); 
    //         // printf("mmap_array[%d] = %d\n", i, mmap_array[i]); 
    //     }
    //     munmap(shared_array, MEM_SIZE);
    //     return 0;
    // } else {
    //     // Parent process
    //     int status;
    //     waitpid(pid, &status, 0);
    //     // if (WIFEXITED(status)) {
    //     //     printf("Child process exited with status: %d\n", WEXITSTATUS(status));
    //     // } else {
    //     //     printf("Child process terminated abnormally\n");
    //     // }
    // }
    // return 0;
    // close(shm_fd);
    // delete[] mmap_array;
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
//     end = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count());
//     printf("Time taken malloc, %ld \n", end);
//     printf("malloc_array[0] = %d\n", malloc_array[0]);
//     return 0;
}