#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

double getInput(){
    double res;
    printf("Number: ");
    fflush(stdout);
    scanf ("%lf ", &res);
    return (double)(res);
}

int main(int argc, char* argv[]){
    double n;
    double sum=0;
    int csize, prank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    srand(time(0));

    int highest_value = 10;
    int vec_size = 1000;
    double x[vec_size], y[vec_size];
    
    if(prank == 0){

        n = getInput();

        for(int i=0; i<n; ++i){
            x[i] = rand()%highest_value;
            y[i] = rand()%highest_value;

            printf("x[%d] = %f\n", i, x[i]);
            printf("y[%d] = %f\n", i, y[i]);
        }
    }
    MPI_Bcast (&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast (y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double s = MPI_Wtime();

    int base_chunk = floor(n/csize);
    int final_chunk = n - (csize-1)*base_chunk;
    int chunk_choice = base_chunk;
    
    if(prank == (csize - 1))
        chunk_choice = final_chunk;
    
    for(int i=0; i < chunk_choice; ++i){
        sum += x[i+base_chunk*prank]*y[i+base_chunk*prank]; 
        //printf("rank: %d index = %d x[index] = %f\n", prank, (i+base_chunk*prank), x[i+base_chunk*prank]);
    }
    printf("rank: %d sum = %f\n", prank, sum);

    double tsum;
    MPI_Reduce (&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double e = MPI_Wtime();
    double d = e - s;
    double mind;
    MPI_Reduce(&d, &mind, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(prank == 0){
        printf("dot product = %f \n", tsum);
        printf("Elapsed time: %f \n", d);
    }

    MPI_Finalize();
    return 0;
}