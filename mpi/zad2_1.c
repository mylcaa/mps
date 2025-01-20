#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

const int MAX_STRING = 100;

int main(void)
{
	char gret[MAX_STRING];
	int csize;
	int prank;
	
	srand(time(0));
	
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &csize);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	
	int rand_val[csize];
	
	for(int i = 0 ; i<csize ; ++i)
		{
			if(i != prank)
			{
				int a = 10*prank + rand()%10;
				MPI_Send(&a, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
		
	for(int i = 0; i < csize ; ++i)
	{
		if(i != prank)
		{
			MPI_Recv(&rand_val[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	
	
	if(prank != 0)
	{
		sprintf(gret, "\nProcess %d recieved: ", prank);
	
		for(int i = 0 ; i<csize ; ++i)
		{
			 if(prank != i)
			 {
				 char num_str[MAX_STRING];
		    		 sprintf(num_str, " %d", rand_val[i]);
		    		 strcat(gret, num_str);
			 }
		} 
		MPI_Send(gret, strlen(gret)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
	else 
	{	
	
		for(int q = 1; q<csize; ++q)
		{
			MPI_Recv(gret, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%s\n", gret);
			
		}
	}	

	MPI_Finalize();
	return 0;
}	

