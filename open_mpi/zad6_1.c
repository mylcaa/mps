#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char* argv[]){
    
    int tc = strtol(argv[1], NULL, 10);
    int n = 0;

    printf("Number: ");
    scanf("%d", &n);

    int biggest_chunksize = floor(n/tc);

    bool prime1[n], prime2[n], prime3[n];
    for(int i = 0; i < n; i++){
        prime1[i] = 1;
        prime2[i] = 1;
        prime3[i] = 1;
    }

    double static_1, static_chunk, dynamic_1;
    double s = omp_get_wtime();
    
    #pragma omp parallel for schedule (static , 1)
    for(int i = 3; i <= n; i++){
        for(int j = 2; j <= floor(sqrt(i)); j++)
            if((i % j)==0){prime1[i] = 0; break;}
    }

    static_1 = omp_get_wtime() - s;
    s = omp_get_wtime();

    #pragma omp parallel for schedule (static , biggest_chunksize)
    for(int i = 3; i <= n; i++){
        for(int j = 2; j <= floor(sqrt(i)); j++)
            if((i % j)==0){prime2[i] = 0; break;}
    }

    static_chunk = omp_get_wtime() - s;
    s = omp_get_wtime();

    #pragma omp parallel for schedule (dynamic , 1)
    for(int i = 3; i <= n; i++){
        for(int j = 2; j <= floor(sqrt(i)); j++)
            if((i % j)==0){prime2[i] = 0; break;}
    }

    dynamic_1 = omp_get_wtime() - s;

    FILE *f = fopen("prime.txt ", "w");
    for(int i = 2; i < n; i++)
        if(prime2[i]) fprintf(f, "%d ", i);
    fclose(f);

    printf("Time static chunksize 1: %lf\n", static_1);
    printf("Time static chunksize n/8: %lf\n", static_chunk);
    printf("Time dynamic chunksize 1: %lf\n", dynamic_1);

    return 0;
}