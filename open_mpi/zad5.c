#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]){
    
    int tc = strtol(argv[1], NULL, 10);
    int n = 0;
    int sum = 0;
    printf("Number: ");
    scanf("%d", &n);

    double s = omp_get_wtime();
    
    #pragma omp parallel for num_threads(tc) reduction(+: sum)
    for(int i = 1; i <= n; i++)
        sum += i;
    
    s = omp_get_wtime() - s;

    printf("\n Sum is %d \n", sum);
    printf("Executed for %lf s \n", s);
    return 0;
}