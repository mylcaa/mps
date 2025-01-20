#include <stdio.h>
#include <math.h>
#include <mpi.h>

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
    
    if(prank == 0){
        n = getInput();
    }
    
    // MPI_Barrier ( MPI_C OMM_WO RLD );
    MPI_Bcast (&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double s = MPI_Wtime();
    double ds = (double)csize;

    int base_chunk = floor(n/csize);
    int final_chunk = n - (csize-1)*base_chunk;
    int chunk_choice = base_chunk;

    if(prank == (csize - 1))
        chunk_choice = final_chunk;
    
    printf("rank: %d sum = %f\n", prank, sum);
    for(int i=1; i<=chunk_choice; ++i){
        sum += i + prank*base_chunk;    
    }
    printf("rank: %d sum = %f\n", prank, sum);

    double tsum;
    MPI_Reduce (&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double e = MPI_Wtime();
    double d = e - s;
    double mind;
    MPI_Reduce(&d, &mind, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(prank == 0){
        printf("Sum first %f integer is %f \n", n, tsum);
        printf("Elapsed time: %f \n", d);
    }

    MPI_Finalize();
    return 0;
}