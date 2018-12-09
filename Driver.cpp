// Jake Thurman ~ COMP 322.A ~ Jacobi
// Runs Jacobi Iterations and Creates a PPM output file

/*
	---------------------------------------------------------
	For documentation on using the Jacobi class, see Jacobi.h
	---------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

// Threshold constant, the iterations stop when changes are smaller than this. 
#define EPSILON 1e-2

// Size of the plate
#define ROWS 1000
#define COLS 1500

// Initial temp
#define INIT_TEMP 50

// Boundry conditions
#define INIT_NORTH 100
#define INIT_SOUTH 0
#define INIT_EAST 100
#define INIT_WEST 0

// Output Config
#define PIXELS_PER_LINE 5
\
/*

Jacobi
- "slices" 3.4.6
- MPI_Scatter, MPI_Scatterv
- Splits up data into num_procs chunks
- Send/Receive top and bottom borders as "constant" rows for neighboring chunks (except on first and last chunk of course)

- MPI_Reduce, MPI_allreduce (allreduce sends the answer to everyone)
 
 */

double* new_jacobi(int maxRows, int maxCols, int initTemperature) {
	// Generate a 1d array to hold our 2d array
	double* arr = (double*)malloc(maxRows * maxCols * sizeof(double));

	// Create all the rows
	for (int i = 0; i < maxRows * maxCols; i++) {
		arr[i] = initTemperature;
	}

	return arr;
}

double* copy_arr(int size, const double* cpy) {
	// Generate a 1d array to hold our 2d array
	double* arr = (double*)malloc(size * sizeof(double));

    // Create all the rows
	for (int i = 0; i < size; i++) {
		arr[i] = cpy[i];
	}

    return arr;
}

double getCell(double* arr, int maxCols, int row, int col) {
	return arr[row * maxCols + col];
}

void setCell(double* arr, int maxCols, int row, int col, double value) {
	arr[row * maxCols + col] = value;
}

double doOnePass(double* in, double* out, int maxr, int maxc) {
	double maxChange = 0; //Used to store the biggest change seen
	double newVal, currChange; //Used for the the processing/swap

#pragma omp parallel \
	default(none) \
	shared(maxChange, in, out, maxc, maxr)
	{
		double newVal, currChange; //Used for the the processing/swap
		double localMaxChange = 0;

#pragma omp for nowait
		for (int r = 1; r < maxr - 1; r++) { //Internal rows
			for (int c = 1; c < maxc - 1; c++) { //Internal columns
				newVal = (getCell(in, maxc, r - 1, c) + getCell(in, maxc, r + 1, c) + getCell(in, maxc, r, c + 1) + getCell(in, maxc, r, c - 1)) / 4.0;
				currChange = getCell(in, maxc, r, c) - newVal;

				if (currChange < 0) {
					currChange *= -1;
				}

				// Write the new result
				setCell(out, maxc, r, c, newVal);

				// Check if this is the max change
				if (currChange > localMaxChange)
					localMaxChange = currChange;
			}
		}

#pragma omp critical
		{
			if (localMaxChange > maxChange)
				maxChange = localMaxChange;
		}
	}

	return maxChange;
}

void toPPM(double time, double* arr) {
	int red, blue; //store the color representation of the heat
	int pixelCount = 0; //counts the number of pixels per line in the file so we don't overflow

	// Include the header for the ppm file
	printf(
        "P3\n%d %d # Time: %f\n# Thurman,Calvis,Oberlander ~ COMP 322.A\n# Epsilon: %f\n255 #max pixel value\n",
		COLS, ROWS, time, EPSILON);
        
	// Calculate/print a color
	for (int i = 0; i < ROWS * COLS; i++) {
		// Get the temp (0 to 100) converted to a 0 to 255 scale
		red = (int)ceil(arr[i] * 255.0 / 100.0);;
		blue = 255 - red;

		// Output to file as "R G B"
		printf("%d 0 %d ", red, blue);

		// Insert a newline after every few pixels so we can 
		//  can be sure we don't exceed 70 characters per line
    	if (++pixelCount % PIXELS_PER_LINE == 0)
			printf("\n");
	}
}

//main
//    takes no parameters
//    returns 0 on success
int main() {
	//0.0 Variable dictionary
	long iterations = 0; //Counter to test when to print
	int my_rank, num_procs;  //holds the rank of each thread and the total number of processes
	int their_rank; //this variable is used to send the id of the slave threads back and forth

	int localRows; // Used to hold the size of the jacobi we're working with on this proc
	double *localJacobi, *masterJacobi; // The Jacobi instance we'll work with and the overall
	double lastMaxChange = EPSILON + 1; //Holds the max change value of the last iteration
	int hasTopRowExtra, hasBottomRowExtra, extraRows; //Used during transmission of neighboring rows

	//1.0 Initialization
	// set up MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	double startTime, stopTime, elapsedSeconds;

	// Set all to use 4 threads
	//omp_set_num_threads(4);

	// Let master init the jacobi instance
	if (my_rank == 0) {
		masterJacobi = new_jacobi(ROWS, COLS, INIT_TEMP);

		// Print basic program info/description
		//printf("Calvis, Oberlander, Thurman ~ COMP 322.A ~ Jacobi\nRuns Jacobi Iterations and Creates a PPM output file\n\n");

		startTime = MPI_Wtime();

		// Init the boundries in a constants based way
		for (int c = 0; c < COLS; c++) {
			setCell(masterJacobi, COLS, 0, c, INIT_NORTH);
			setCell(masterJacobi, COLS, ROWS - 1, c, INIT_SOUTH);
		}
		for (int r = 0; r < ROWS; r++) {
			setCell(masterJacobi, COLS, r, 0, INIT_WEST);
			setCell(masterJacobi, COLS, r, COLS - 1, INIT_EAST);
		}
	}

	// Send out mopst of what we need using scatter
	localRows = ROWS / num_procs;
	localJacobi = (double*)malloc(localRows * COLS * sizeof(double));

	MPI_Scatter(masterJacobi, localRows * COLS, MPI_DOUBLE, localJacobi, localRows * COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Handle trading off top and bottom rows
	hasTopRowExtra = my_rank != 0;
	hasBottomRowExtra = my_rank != (num_procs - 1);
	extraRows = hasTopRowExtra + hasBottomRowExtra;
	
	// Create a local jacobi with neighboring rows. 
	//  First, we copy over what we have so far.
	double* withNeighbors = (double*)malloc((extraRows + localRows) * COLS * sizeof(double));
	
	for (int i = 0; i < localRows * COLS; i++) {
		withNeighbors[i + (hasTopRowExtra * COLS)] = localJacobi[i];
	}

	free(localJacobi);
	localJacobi = withNeighbors;
	withNeighbors = NULL;
	

	//2.0 Running Jacobi Iterations
	while (lastMaxChange >= EPSILON) {
		// Sync up neighbors again!!!
		if (hasTopRowExtra) {
			// Send first row I own to neighbor above
			MPI_Send(localJacobi + COLS, COLS, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
		}

		if (hasBottomRowExtra) {
			// Recieve neighbor below's first row owned as overall last row
			MPI_Recv(localJacobi + ((localRows + hasTopRowExtra) * COLS), COLS, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, NULL);

			// Send last row to neighbor below
			MPI_Send(localJacobi + ((localRows + hasTopRowExtra - 1) * COLS), COLS, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
		}

		if (hasTopRowExtra) {
			// Recieve neighbor above's last row owned as overall first row
			MPI_Recv(localJacobi, COLS, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, NULL);
		}

		// Load up the next jacobi iteration and store it
		double* next = copy_arr((localRows + extraRows) * COLS, localJacobi); // Stores the jacobi instance we perform a pass to.
		double localLastMaxChange = doOnePass(localJacobi, next, localRows + extraRows, COLS);

		// Sync the last max change made
		MPI_Allreduce(&localLastMaxChange, &lastMaxChange, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

		// Set this as our new current instance
		free(localJacobi);
		localJacobi = next;
		iterations++;
	}

	// Reduce the data back into the master...
	int toSendSize = localRows * COLS;
	MPI_Gather(localJacobi + (hasTopRowExtra * COLS), toSendSize, MPI_DOUBLE, masterJacobi, toSendSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//3.0 Outputing results
	if (my_rank == 0) {
		// Print runtime
		stopTime = MPI_Wtime();
		elapsedSeconds = stopTime - startTime;
		//printf("Time to completion:   %f\n", elapsedSeconds);

		// Create file
		toPPM(elapsedSeconds, masterJacobi);

		// Exit
		//printf("Program completed successfully.\n");
	    
		free(masterJacobi);
	}

	//4.0 Cleanup 
	free(localJacobi);
	MPI_Finalize();

	return 0; //no error
}
