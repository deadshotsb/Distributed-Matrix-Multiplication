/* The program is submitted by Sombit Bose(22CS60R31) for Assignment 7 of CL Lab.
The approach is to use a Server and Clinet interface to perform multi-thread matrix 
multiplication, The program takes input of command line args -
1. switch = -s for server and -c for client
2. ip: the IP address on which to listen to in case of server, and connect to
in case of client.
3. port number = port number
4. Number of clients: only in case of a server.
5. Path to local file with job: only in case of a server.

Server Side-

It reads the set of matrices from the input file and creates a socket for communication,
binds the socket with the socket address ad enters a listen state for clients to send a request.
The Server accepts a client request and interacts with the client using threads to send and recieve
informations, After recieving the evaluated matrix from a batch of matrices, clinet sends a result
to the server, it stores it in a data structure and later after joining the threads, it calls the
multi-thread matrix multiplication to finally evaluate the matrix Y.

Client Side - 

It creates a socket and connects to the server via that socket, the server sends a batch of matrices
to the client. It evaluates the matrix multiplication result using multi-threaded matrix multiplication
and returns the result to the server.
*/

#include <pthread.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define SIZE 100

//structure utilized in the matrix multiplication thread
typedef struct matrix_mul {
	int ***mat_chain;    //batch of matrix 
	int *rows;         //input dimensions
	int *cols;
	int ***mat_res;           //resultant matrix list
	int *rows_res;        //output dimensions
	int *cols_res;
	int m;            //batch_size
}MAT_DATA;

//structure that is used to return a single matrix from a function
typedef struct matrix_result{
	int **matrix;     //matrix
	int r;        //dimensions
	int c;
}mat_res;

//structure used in the server client messaging thread
typedef struct serv_cli_thd {
	int ***matrix_collection;   //collection of matrices
	int *num_rows;            //dimensions
	int *num_cols;
	int sockfd;                //opened socket by the server
	int *nsockfd_list;              //list sockets for communications by the m clients
	struct sockaddr_in slave_addr;
	struct sockaddr_in master_addr;
	int n;                //total number of matrix
	int m;               // total number of clients
	mat_res *res;
} THD_DATA;

//global variables defined to be used by the threads
THD_DATA master_thd;
MAT_DATA mat_thread_data;
mat_res temp_ret;

/*
mat_mul - This function is utilized by the threads of matrix multiplication function, where 
it returns a matrix multiplication of two matrix using the argument i.
Argument - args - void pointer that is passed through the thread function.
Returns - void pointer denoting the termination of the thread
*/
void *mat_mul(void *args) {
	int i, j, k, l, a, b, r, c, temp;
	int **A, **B, **C;
	i = (int)args;
	a = 2*i;
	b = 2*i+1;
	//printf("%d %d\n", a, b);
	if(b>=mat_thread_data.m) {           // if a is only the last matrix in the list.
		r = mat_thread_data.rows[a];
		c = mat_thread_data.cols[a];		
		C = (int **)calloc(r, sizeof(int *));
		for(j=0; j<r; j++) {
			C[j] = (int *)calloc(c, sizeof(int));
			for(k=0; k<c; k++)
				C[j][k] = mat_thread_data.mat_chain[a][j][k];
		}
		mat_thread_data.mat_res[i] = C;
		mat_thread_data.rows_res[i] = mat_thread_data.rows[a];
		mat_thread_data.cols_res[i] = mat_thread_data.cols[a];
	}
	else {                         // a thread multiplies the a,b matrix
		A = mat_thread_data.mat_chain[a];
		B = mat_thread_data.mat_chain[b];
		r = mat_thread_data.rows[a];
		c = mat_thread_data.cols[b];
		temp = mat_thread_data.cols[a];
		C = (int **)calloc(r, sizeof(int *));
		for(j=0; j<r; j++) {
			C[j] = (int *)calloc(c, sizeof(int));
			for(k=0; k<c; k++) {
				C[j][k] = 0;
				for(l=0; l<temp; l++)
					C[j][k] = C[j][k] + (A[j][l] * B[l][k]);
			}
		}
		mat_thread_data.mat_res[i] = C;
		mat_thread_data.rows_res[i] = r;
		mat_thread_data.cols_res[i] = c;
	}
	pthread_exit((void *) 0);        //successful termination of the thread
}

/*
server_thread() - This function is utilized by the server threads to communicate through to the client,
It calculates the batch to send to the client and sends the batch of matrices to the client. The client 
processes the data and returns the matrix multiplication result the server. This function stores 
the result in an appropriate data structure for further processing.
Argument - args - void pointer that is passed through the thread function.
Returns - void pointer denoting the termination of the thread
*/
void *server_thread(void *args) {
	int i, batch_size, st, sp, j, k, loop_in, loop1, m, n, r, c;
	n = master_thd.n;
	m = master_thd.m;
	batch_size = n/m;
	i = (int)args;
	int rem = n%m;
	// calculating the batch size for each server thread.
	if (rem != 0) {
		if (i <= (rem-1)) {
			st = i*batch_size + i;
			sp = st + batch_size + 1;
		}
		else {
			st = i*batch_size + rem;
			sp = st + batch_size;
		}
	}
	else {
		st = i*batch_size;
		sp = st + batch_size;
	}
	char buff[SIZE];
	char *temp;
	char buff_temp[SIZE];
	//close(master_thd.sockfd); 
	for(loop_in=0; loop_in < SIZE; loop_in++) 
		buff[loop_in] = '\0'; // Initialize buffer 
	sprintf(buff, "%d", sp - st);
	send(master_thd.nsockfd_list[i], buff, SIZE, 0); 
	printf("Matrices sent to Client %d\n\n", (i+1));
	for(j=st; j<sp; j++) {
		for(loop_in=0; loop_in < SIZE; loop_in++) 
			buff[loop_in] = '\0';
		sprintf(buff, "%d %d", master_thd.num_rows[j], master_thd.num_cols[j]); 
		printf("\nDimension = %s\n", buff);
		send(master_thd.nsockfd_list[i], buff, SIZE, 0);
		printf("Matrix = \n\n"); 
		for(k=0; k<master_thd.num_rows[j]; k++) {
			for(loop_in=0; loop_in < SIZE; loop_in++) 
				buff[loop_in] = '\0'; // Initialize buffer 
			for(loop1 = 0; loop1<(master_thd.num_cols[j]-1); loop1++) {
				sprintf(buff_temp, "%d ", master_thd.matrix_collection[j][k][loop1]);
				strcat(buff, buff_temp);
			}
			sprintf(buff_temp, "%d ", master_thd.matrix_collection[j][k][master_thd.num_cols[j]-1]);
			strcat(buff, buff_temp);
			printf("%s\n", buff);
			send(master_thd.nsockfd_list[i], buff, SIZE, 0); // Send value of each row of a matrix one by one
		}
	}
	printf("\nServer has sent the message to client\n");
	printf("\nMatrix Recieved\n");
	printf("\nResultant Matrix Recieved from Client %d\n", (i+1));
	for(loop_in=0; loop_in < SIZE; loop_in++) 
		buff[loop_in] = '\0'; // Initialize buffer 
	recv(master_thd.nsockfd_list[i], buff, 100, 0); // Receive message 
	sscanf(buff, "%d %d", &r, &c);
	printf("Dimension = %s\n", buff);
	master_thd.res[i].r = r;
	master_thd.res[i].c = c;
	printf("Matrix = \n\n");
	for(j=0; j<r; j++) 
		master_thd.res[i].matrix = (int **)calloc(r, sizeof(int *));
	char *tok;
	for(j=0; j<r; j++) {
		master_thd.res[i].matrix[j] = (int *)calloc(c, sizeof(int));
		for(loop_in=0; loop_in < SIZE; loop_in++)
			buff[loop_in] = '\0'; // Initialize buffer 
		recv(master_thd.nsockfd_list[i], buff, 100, 0); // Receive message 
		printf("%s\n", buff);
		tok = strtok(buff, " ");
		for(k=0; k<c; k++) {
			master_thd.res[i].matrix[j][k] = atoi(tok);
			tok = strtok(NULL, " ");
		}
	}
	printf("\n\n");
	close(master_thd.nsockfd_list[i]);
	pthread_exit((void *) 0);
}

/*
mat_res_multi_threaded_mat_mul() - this function accepts a collection of matrix
and calculates their product by using multi-threaded matrix multiplication 
technique as described in the problem.
Arguments - mat_collections - array of mat_res stuctures that contains the matrix informations
	    n - number of matrices
Return - product of the n matrices
*/
mat_res multi_threaded_mat_mul(mat_res *mat_collections, int n) {
	int i, j, k;
	void *status;
	//defining the variables in mat_thread_data global structure
	mat_thread_data.mat_chain = (int ***)calloc(n, sizeof(int **));
	mat_thread_data.mat_res = (int ***)calloc(n, sizeof(int **));
	mat_thread_data.rows = (int *)calloc(n, sizeof(int));
	mat_thread_data.cols = (int *)calloc(n, sizeof(int));
	mat_thread_data.rows_res = (int *)calloc(n, sizeof(int));
	mat_thread_data.cols_res = (int *)calloc(n, sizeof(int));	
	for(i=0; i<n; i++) {
		mat_thread_data.mat_chain[i] = mat_collections[i].matrix;
		mat_thread_data.rows[i] = mat_collections[i].r;
		mat_thread_data.cols[i] = mat_collections[i].c; 
	}
	mat_thread_data.m = n;
	int ***temp1, *temp2;
	while(n>1){
		pthread_t thd_no[(n+1)/2];    //creating (n+1)/2 threads for each steps
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		for(i=0; i<((n+1)/2); i++) {
			pthread_create(&thd_no[i], &attr, mat_mul, (void *)i);
		}
		pthread_attr_destroy(&attr);
		for(i=0; i<((n+1)/2); i++)
			pthread_join(thd_no[i], &status);
		temp1 = mat_thread_data.mat_chain;          //utilizing the input matrix array for next itertion
		mat_thread_data.mat_chain = mat_thread_data.mat_res;
		mat_thread_data.mat_res = temp1;
		temp2 = mat_thread_data.rows;          //utilizing the input matrix array rows for next itertion
		mat_thread_data.rows = mat_thread_data.rows_res;
		mat_thread_data.rows_res = temp2;
		temp2 = mat_thread_data.cols;          //utilizing the input matrix array columns for next itertion
		mat_thread_data.cols = mat_thread_data.cols_res;
		mat_thread_data.cols_res = temp2;
		mat_thread_data.m = (n+1)/2;
		//pthread_exit(NULL);
		n = (n+1)/2;
	}
	temp_ret.matrix = mat_thread_data.mat_chain[0];           //result present in input due to last iteration swap of input and output matrix.
	temp_ret.r = mat_thread_data.rows[0];
	temp_ret.c = mat_thread_data.cols[0];
	return temp_ret;           //returning the result matrix
}

/*
check_ip() - validates the input port number if it is in valid valid range.
Arguments- p - charecter array denoting port number
Return - 1 if correct port entered, 0 otherwise
*/
int check_port(char *p) {
	int pos = 0;
	while(p[pos] != '\0') {
		if (!isdigit(p[pos]))    //checking for digits in port number
			return 0;
		pos ++;
	}
	int port = atoi(p);
	if (port<1024) {       // port 0 to 1023 are pre-defined for applications
		printf("Port cannot be used\n");
		return 0;
	}
	return 1;
}

/*
check_num() - validates the number of clients entered if it is a number.
Arguments- n - charecter array denoting number input
Return - 1 if correct number entered, 0 otherwise
*/

int check_num(char *n) {
	int pos = 0;
	while(n[pos] != '\0') {
		if (!isdigit(n[pos]))    //checking if it is a number
			return 0;
		pos ++;
	}
	return 1;
}

/*
check_ip() - validates the input ip address if it is in valid IPv4 dotted
decimal format.
Arguments- ip - charecter array denoting the ip address
Return - 1 if correct ip entered, 0 otherwise
*/
int check_ip(char *ip) {
	int pos = 0, num = 0, dot = 0;
	while(ip[pos] != '\0') {      //traversing the array
		if (ip[pos] == '.') {   //checking for '.'
			dot ++;
		}
		else if (isdigit(ip[pos])) {       //checking for number
			if(ip[pos+1] == '.' || ip[pos+1] == '\0')
				num ++;
		}
		else 
			return 0;
		pos++;
	}
	if(num == 4 && dot == 3)         // ip = a.b.c.d exploiting this property for check
		return 1;
	else
		return 0;
}

int main(int argc, char **argv) {
	
	char ip_address[SIZE];
	if (check_ip(argv[2]) == 0){    //validating the ip address
		printf("Wrong Input\n");
		exit(0);
	}
	
	if (check_port(argv[3]) == 0){       // validating the port number to be a valid port
		printf("Wrong Input\n");
		exit(0);
	}
	
	int port_no = atoi(argv[3]);
	strcpy(ip_address, argv[2]);
	if(strcmp(argv[1], "-s") == 0 || strcmp(argv[1], "-S") == 0) {  // Master(Server)
		FILE *inp;
		if (check_num(argv[4]) == 0) {     // validating the number of clients entered
			printf("Wrong Input\n");
			exit(0);
		}
		int n, i, j, k, r, c, m = atoi(argv[4]);
		int ***matrix_collection;
		inp = fopen(argv[5], "r");    //opening the file given
		char buff[SIZE];
		fgets(buff, SIZE, inp);
		int pos = 0;
		n = 0;
		while(buff[pos] != '\n' && buff[pos] != '\0') {
			if ((int)buff[pos]>=48 && (int)buff[pos]<=58) 
				n = n*10 + ((int)buff[pos] - 48);
			else {
				printf("Wrong Input\n");
				exit(0);
			}
			pos++;
		}
		if (n<m) {   //error handling for n and m.
			printf("Number of clients must be less or equal to number of matrices\n");
			exit(0);
		}
		int *num_rows, *num_cols;
		num_rows = (int *)calloc(n, sizeof(int));
		num_cols = (int *)calloc(n, sizeof(int));
		matrix_collection = (int ***)calloc(n, sizeof(int **));
		for(i=0; i<n; i++) {
			fgets(buff, SIZE, inp);
			r = 0, c = 0;
			pos = 0;
			//for extracting the rows and columns
			while(buff[pos] != ' ' && buff[pos] != '\n' && buff[pos] != '\0') {
				if ((int)buff[pos]>=48 && (int)buff[pos]<=58) 
					r = r*10 + ((int)buff[pos] - 48);
				else {
					printf("Wrong Input\n");
					exit(0);
				}
				pos++;
			}
			if (buff[pos] == '\n' && buff[pos] == '\0') {
				printf("Wrong Input\n");
				exit(0);
			}
			pos++;
			while(buff[pos] != '\n' && buff[pos] != '\0') {
				if ((int)buff[pos]>=48 && (int)buff[pos]<=58) 
					c = c*10 + ((int)buff[pos] - 48);
				else {
					printf("Wrong Input\n");
					exit(0);
				}
				pos++;
			}
			matrix_collection[i] = (int **)calloc(r, sizeof(int *));
			num_rows[i] = r;
			num_cols[i] = c;
			// for extracting the matrix values
			if(i > 0 && num_rows[i] != num_cols[i-1]) {
				printf("Matrix multiplication incompatible\n");
				exit(0);
			}
			for(j=0; j<r; j++) {
				matrix_collection[i][j] = (int *)calloc(c, sizeof(int));
				int nums = 0, k=0, blank = 0, val = 0, sign_flag = 0;
				pos = 0;
				fgets(buff, SIZE, inp);
				//printf("%s\n", buff);
				while(buff[pos] != '\0' && buff[pos] != '\n') {
					if(buff[pos] == ' ') {
						blank ++;
						if (sign_flag == 1)            // for negative values
							matrix_collection[i][j][k] = -1 * val;
						else 
							matrix_collection[i][j][k] = val;
						val = 0;
						k++;
					}
					else if ((int)buff[pos]>=48 && (int)buff[pos]<=58) {         //check for numbers      
						val = val*10 + ((int)buff[pos] - 48);
						if (pos == 0 || buff[pos-1] == ' ' || buff[pos-1] == '-')
							nums++;
					}
					else if (buff[pos] == '-') {                 // for negative sign
						if ((int)buff[pos+1] >= 48 && (int)buff[pos+1] <= 58) {
							if(pos == 0 || (pos>0 && buff[pos-1] == ' '))
								sign_flag = 1;
							else {
								printf("Wrong Input\n");
								exit(0);
							}
						}
						else {
							printf("Wrong Input\n");
							exit(0);
						}
					}
					else {
						printf("Wrong Input\n");
						exit(0);
					}
					pos ++;
				}
				if (blank == (c-1) && nums == c) {
					if (sign_flag == 1)
						matrix_collection[i][j][c-1] = -1 * val;
					else
						matrix_collection[i][j][c-1] = val;
				}
				else {
					printf("Wrong Input\n");
					exit(0);
				}
			}
		}
		fclose(inp);
		printf("Input Matrix Read Succesfully\n");
		/*for(i = 0; i<n; i++) {
			for(j =0; j<num_rows[i]; j++) {
				for(k=0; k<num_cols[i]; k++)
					printf("%d ", matrix_collection[i][j][k]);
				printf("\n");
			}
			printf("\n");
		}*/
		int sockfd, newsockfd; // Socket descriptors
		int slavelen;
		struct sockaddr_in master_addr;
		i=0;
		char buf[100];    // this buffer is used for communication
		if ((sockfd = socket(AF_INET, SOCK_STREAM, 0) ) < 0) {  // creating socket using internet protocol
			printf("Cannot create socket\n");
			exit(0);
		}
		master_addr.sin_family = AF_INET;
		master_addr.sin_addr.s_addr = inet_addr(ip_address);
		master_addr.sin_port = port_no;
		if ( bind(sockfd, (struct sockaddr *) &master_addr, sizeof(master_addr)) < 0) { //we bind the server using master_addr
			printf("Unable to bind local address\n");
			exit(0);
		}
		listen(sockfd, m);
		mat_res *res = (mat_res *)calloc(m, sizeof(mat_res));
		void *status;
		master_thd.nsockfd_list = (int *)calloc(m, sizeof(int));
		pthread_t serv_thd_no[m];    // creating a thread for m clients 
		pthread_attr_t attr;        // attributes for thread
		pthread_attr_init(&attr);     // initializing the threads
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);    //joinable threads
		for(i=0; i<m; i++) {
			struct sockaddr_in slave_addr;
			slavelen = sizeof(slave_addr);
			newsockfd = accept(sockfd, (struct sockaddr *) &slave_addr, &slavelen);  //servers waits to accept request from client 
			if (newsockfd < 0) {
				printf("Accept error\n");
				exit(0);
			}
			else {
				printf("Request from %d client\n", (i+1));
			}
			
			// start thread from here
			// providing values to the global structures for using them in the thread
			master_thd.matrix_collection = matrix_collection;
			master_thd.num_rows = num_rows;
			master_thd.num_cols = num_cols;
			master_thd.sockfd = sockfd;
			master_thd.nsockfd_list[i] = newsockfd;
			master_thd.master_addr.sin_family = master_addr.sin_family;
			master_thd.master_addr.sin_addr.s_addr = master_addr.sin_addr.s_addr;
			master_thd.master_addr.sin_port = master_addr.sin_port;
			master_thd.slave_addr.sin_family = slave_addr.sin_family;
			master_thd.slave_addr.sin_addr.s_addr = slave_addr.sin_addr.s_addr;
			master_thd.slave_addr.sin_port = slave_addr.sin_port;
			master_thd.n = n;
			master_thd.m = m;
			master_thd.res = res;
			pthread_create(&serv_thd_no[i], &attr, server_thread, (void *)i);   //creating the thread using parameters
			//close(newsockfd);
		}
		pthread_attr_destroy(&attr);
		for(i=0; i<m; i++)
			pthread_join(serv_thd_no[i], &status);   //joining the created threads
		close(sockfd);
		printf("Final Evaluated Matrix\n\n");
		mat_res Y = multi_threaded_mat_mul(res, m);       //calculating the Y matrix from m recieved matrices
		printf("Dimensions = %d X %d\n", Y.r, Y.c);
		printf("Matrix = \n\n");
		for(i=0; i<Y.r; i++) {
			for(j=0; j<Y.c; j++)
				printf("%d ", Y.matrix[i][j]);
			printf("\n");
		}
		//free(Y);
		pthread_exit(NULL);
		//call multi-threaded matrix multiplication.
	}
	else if (strcmp(argv[1], "-c") == 0 || strcmp(argv[1], "-C") == 0) {   //Slave(Client)
		int sockfd;
		char *temp;
		char buff_temp[SIZE];
		struct sockaddr_in slave_addr;
		int i, j, k, batch_size, r, c, loop_in;
		char buff[100];      //buffer used for communication
		if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {          // creating the socket
			printf("Unable to create socket\n");
			exit(0);
		}
		slave_addr.sin_family = AF_INET;
		slave_addr.sin_addr.s_addr = inet_addr(ip_address);   
		slave_addr.sin_port = port_no;
		
		if (connect(sockfd, (struct sockaddr *) &slave_addr, sizeof(slave_addr)) < 0) {  //client connects to server using connect()
			printf("Unable to connect to server\n");
			exit(0);
		}
		
		for(loop_in=0; loop_in < 100; loop_in++) 
			buff[loop_in] = '\0';
		recv(sockfd, buff, 100, 0);
		printf("Number of Matrix scheduled to send %s\n", buff);
		printf("\n\nFollowing MAtrices have been recieved \n\n");
		sscanf(buff, "%d", &batch_size);
		mat_res *mat_recv = (mat_res *)calloc(batch_size, sizeof(mat_res));
		for(k =0; k< batch_size; k++) {
			for(loop_in=0; loop_in < 100; loop_in++) 
				buff[loop_in] = '\0';
			recv(sockfd, buff, 100, 0);
			sscanf(buff, "%d %d", &r, &c);             //recieving rows and cols of a matrix
			printf("Dimension = %s\n", buff);
			mat_recv[k].r = r;
			mat_recv[k].c = c;
			printf("Matrix = \n\n");
			mat_recv[k].matrix = (int **)calloc(r, sizeof(int *));
			for(i=0; i<r; i++) {
				mat_recv[k].matrix[i] = (int *)calloc(c, sizeof(int));
				for(loop_in=0; loop_in < 100; loop_in++) 
					buff[loop_in] = '\0';
				recv(sockfd, buff, 100, 0);            //recieving each rows of the matrix
				printf("%s\n", buff);
				temp = strtok(buff, " ");
				for(j=0; j<c; j++) {
					mat_recv[k].matrix[i][j] = atoi(temp);
					temp = strtok(NULL, " ");         //tokenization and extraction of the values
				}
			}
		}
		
		printf("\nClient Read the information successfully\n");
		
		mat_res y_temp = multi_threaded_mat_mul(mat_recv, batch_size);// multi-thread matrix multiplication and store in mat_res y_temp 
		
		//mat_res *y_temp = &mat_recv[0];
		printf("Resultant Matrix from Client End\n\n");
		for(i=0; i < SIZE; i++)
			buff[i] = '\0';
		sprintf(buff, "%d %d", y_temp.r, y_temp.c);    //sending the rows and cols of the resultant matrix
		send(sockfd, buff, SIZE, 0);
		printf("Dimensions = %s\n", buff);
		for(i=0; i<y_temp.r; i++) {
			for(k=0; k < SIZE; k++) 
				buff[k] = '\0';
			for(j=0; j<(y_temp.c-1); j++) {
				sprintf(buff_temp, "%d ", y_temp.matrix[i][j]);           // putting each row of the resultant matrix to buffer 
				strcat(buff, buff_temp);
			}
			sprintf(buff_temp, "%d", y_temp.matrix[i][y_temp.c - 1]);
			strcat(buff, buff_temp);
			printf("%s\n", buff);
			send(sockfd, buff, SIZE, 0);                     //sending the matrix data to server
		}
		printf("\n\nResultant Matrix Sent to Server\n");
		//free(y_temp);
		close(sockfd);
	}
	else            // error handling for wrong input
		printf("Wrong Input");
	return 0;
}
