#include <stdio.h>
#include <mpi.h>
#include <math.h>

int returnSize(char* fname){
    FILE *f = fopen(fname, "r");
    int dim = 0;
    double tmp;
    while(fscanf(f, "%lf ", &tmp) != EOF)
    dim++;
    fclose(f);
    return dim;
}

double* loadVec(char* fname, int n){
    FILE *f = fopen(fname, "r");
    double *res = new double[n];
    double *it = res;
    while(fscanf(f, "%lf", it++) != EOF);
    fclose (f);
    return res;
}

double* loadMat(char* fname, int n){
    FILE *f = fopen(fname, "r");
    double *res = new double [n*n];
    double *it = res;
    while(fscanf(f, "%lf", it++) != EOF);
    fclose(f);
    return res;
}

void logRes(const char* fname, double* res, int n){
    FILE *f = fopen(fname, "w");
    for(int i = 0; i != n ; ++i)
        fprintf(f, "%lf ", res[i]);
    fclose(f);
}

int main(int argc, char* argv[]){
    int csize;
    int prank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    char *vfname = argv[1];
    char *mfname = argv[2];
    int dim;
    double* mat;
    double* vec;
    double* tmat;
    double* lres;
    double* res;
    if(prank == 0)
        dim = returnSize(vfname);
    
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(prank == 0)
        vec = loadVec(vfname, dim);
    else
        vec = new double[dim];
    
    MPI_Bcast(vec, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(prank == 0)
        tmat = loadMat(mfname, dim);
    
    int to_regular = floor(dim/csize);
    int to_leftover = dim - csize*to_regular;
    int msize_regular = to_regular*dim;
    int msize_leftover = (to_regular + to_leftover)*dim;
    int send_cnt[csize];
    int displ_scatter[csize];
    int recv_cnt[csize];
    int displ_gather[csize];

    if(prank == 0){
        for(int i=0; i<csize; ++i){
            send_cnt[i] = ((i != (csize-1))? msize_regular : msize_leftover); 
            displ_scatter[i] = i*msize_regular;

            recv_cnt[i] = ((i != (csize-1))? to_regular : (to_regular+to_leftover)); 
            displ_gather[i] = i*to_regular;

            printf("recv[%d]=%d displ[%d]=%d\n", i, recv_cnt[i], i, displ_gather[i]);
        }
    }
    mat = new double[msize_leftover];
    
    //MPI_Scatter(tmat, msize, MPI_DOUBLE, mat, msize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(tmat, send_cnt, displ_scatter, MPI_DOUBLE, mat, msize_leftover, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int to = to_regular;
    if(prank == (csize-1))
        to = to_leftover+to_regular;

    lres = new double[to];


    for(int i = 0; i != to; ++i){
        double s = 0;
        for(int j = 0; j != dim; ++j){
            s += mat[i*dim + j]*vec[j];
        }
        lres[i] = s;
    }

    printf("RANK %d to %d \n", prank, to);

    if(prank == 0)
        res = new double[dim];
    
    //MPI_Gather(lres, to, MPI_DOUBLE, res, to, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lres, to, MPI_DOUBLE, res, recv_cnt, displ_gather, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if ( prank == 0){
        logRes("res.txt ", res, dim);
    }
    
    if(prank == 0){
        delete[] tmat;
        delete[] res;
    }

    delete[] vec;
    delete[] mat;
    delete[] lres;
    MPI_Finalize();
    
    return 0;
}