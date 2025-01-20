#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]){
    
    int tc = strtol(argv[1], NULL, 10);
    int n = 0;
    int sum = 0;
    int thread_sums[tc];
    printf("Number: ");
    scanf("%d", &n);

    for(int i = 0; i < tc; i++)
        thread_sums[i] = 0;

    double s = omp_get_wtime();
    
    #pragma omp parallel for num_threads(tc)
    for(int i = 1; i <= n; i++){
        int rank = omp_get_thread_num();
        thread_sums[rank] += i;
    }

    for(int i = 0; i < tc; i++)
        sum += thread_sums[i];

    s = omp_get_wtime() - s;

    printf("\n Sum is %d \n", sum);
    printf("Executed for %lf s \n", s);
    return 0;
}