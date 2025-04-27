#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Sequential Bubble Sort
void bubbleSortSequential(int arr[], int n) {
    int temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Optimized Parallel Bubble Sort
void bubbleSortParallel(int arr[], int n) {
    if (n < 2000) {
        bubbleSortSequential(arr, n);
        return;
    }

    int temp;
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for private(temp)
        for (int j = (i % 2 == 0) ? 0 : 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to generate random numbers
void generateRandomArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int main() {
    srand(time(0));
    omp_set_num_threads(4);

    int inputSizes[5];
    printf("Enter 5 input sizes: ");
    for (int i = 0; i < 5; i++) {
        scanf("%d", &inputSizes[i]);
    }

    printf("\n Rohit Chauhan BE A 41008\n");

    printf("\n+------------+---------------------+----------------------+-------------------+-------------------+\n");
    printf("| Input Size | Bubble Seq Time      | Bubble Par Time       | Bubble Speedup    | Bubble Efficiency |\n");
    printf("+------------+---------------------+----------------------+-------------------+-------------------+\n");

    for (int t = 0; t < 5; t++) {
        int n = inputSizes[t];
        int *arr1 = (int *)malloc(n * sizeof(int));
        int *arr2 = (int *)malloc(n * sizeof(int));

        generateRandomArray(arr1, n);
        for (int i = 0; i < n; i++) {
            arr2[i] = arr1[i];
        }

        double start_time, end_time;

        start_time = omp_get_wtime();
        bubbleSortSequential(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSortSeqTime = end_time - start_time;

        start_time = omp_get_wtime();
        bubbleSortParallel(arr2, n);
        end_time = omp_get_wtime();
        double bubbleSortParTime = end_time - start_time;

        double bubbleSortSpeedup = bubbleSortSeqTime / bubbleSortParTime;
        double bubbleSortEfficiency = bubbleSortSpeedup / 4;

        printf("| %10d | %19.4f | %20.4f | %17.4f | %17.4f |\n",
               n, bubbleSortSeqTime, bubbleSortParTime, bubbleSortSpeedup, bubbleSortEfficiency);

        free(arr1);
        free(arr2);
    }

    printf("+------------+---------------------+----------------------+-------------------+-------------------+\n");
    return 0;
}
