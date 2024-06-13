#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <algorithm>
#include <iterator>
#include <cmath>

#include <string>
#include <fstream>
#include <sstream>

#include "ap_int.h"
#include "Hadi_MatrixMult.h"
#include "kernelmatrixmult.h"

using std::cout;
using std::endl;


#define max_N_adj MAX_N
#define max_M_fea MAX_M
#define max_P_w MAX_P


#define citeseer
#define layer2


#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <algorithm>
#include <iterator>
#include <cmath>

#include <string>
#include <fstream>
#include <sstream>

#include "ap_int.h"
#include "Hadi_MatrixMult.h"
#include "kernelmatrixmult.h"

using std::cout;
using std::endl;


#define max_N_adj MAX_N
#define max_M_fea MAX_M
#define max_P_w MAX_P


#define citeseer
#define layer2


std::string address = "/home/hadi/opt/Workspace/Vitis_HLS/gfades";




// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                     csrDataS              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
#ifdef pubmed

    
    int N_adj = 19717;  // number of nodes
    int M_fea = 500;  // number of input features
    int P_w = 18;  // number of features in the hidden layer
    int NNZ_adj = 108365;  // number of non-zero values of adjacency
    int NNZ_fea = 988031;  // number of non-zero values of feature

    static const std::string adj_name = address +"/csrData/pubmed_adj.txt";
    static const std::string fea_name = address +"/csrData/pubmed_fea.txt";
    static const std::string w_name = address +"/csrData/pubmed_weights.txt";
    

    #ifdef layer2
        int N_adj2 = 19717;  // number of nodes
        int M_fea2 = 18;  // number of input features
        int P_w2 = 3;  // number of features in the hidden layer
        int NNZ_adj2 = 108365;  // number of non-zero values of adjacency
        int NNZ_fea2 = 19717 * 18;  // number of non-zero values of feature

        static const std::string adj_name2 = address +"/csrData/pubmed_adj.txt";
        static const std::string fea_name2 = address +"/csrData/pubmed_fea.txt";
        static const std::string w_name2 = address +"/csrData/pubmed_weights2.txt";
    #endif

  
#endif

#ifdef cora

    
    int N_adj = 2708;  // number of nodes
    int M_fea = 1433;  // number of input features
    int P_w = 64;  // number of features in the hidden layer
    int NNZ_adj = 13264;  // number of non-zero values of adjacency
    int NNZ_fea = 49216;  // number of non-zero values of feature

    static const std::string adj_name = address +"/csrData/cora_adj.txt";
    static const std::string fea_name = address +"/csrData/cora_fea.txt";
    static const std::string w_name = address +"/csrData/cora_weights.txt";
  

    #ifdef layer2
        int N_adj2 = 2708;  // number of nodes
        int M_fea2 = 64;  // number of input features
        int P_w2 = 7;  // number of features in the hidden layer
        int NNZ_adj2 = 13264;  // number of non-zero values of adjacency
        int NNZ_fea2 = 2708 * 64;  // number of non-zero values of feature

        static const std::string adj_name2 = address +"/csrData/cora_adj.txt";
        static const std::string fea_name2 = address +"/csrData/cora_fea.txt";
        static const std::string w_name2 = address +"/csrData/cora_weights2.txt";
    #endif
    
#endif

#ifdef citeseer
    
    int N_adj = 3327;  // number of nodes
    int M_fea = 3703;  // number of input features
    int P_w = 21;  // number of features in the hidden layer
    int NNZ_adj = 12431;  // number of non-zero values of adjacency
    int NNZ_fea = 105165;//105165;  // number of non-zero values of feature

    static const std::string adj_name = address + "/csrData/citeseer_adj.txt";
    static const std::string fea_name = address +"/csrData/citeseer_fea.txt";
    static const std::string w_name = address +"/csrData/citeseer_weights.txt";

   

    #ifdef layer2

        int N_adj2 = 3327;  // number of nodes
        int M_fea2 = 21; //3703;  // number of input features
        int P_w2 = 6; //21;  // number of features in the hidden layer
        int NNZ_adj2 = 12431;  // number of non-zero values of adjacency
        int NNZ_fea2 = 3327*21;//105165;  // number of non-zero values of feature

        static const std::string adj_name2 = address + "/csrData/citeseer_adj.txt";
//        static const std::string fea_name2 = address +"/csrData/citeseer_feat.txt";
        static const std::string w_name2 = address +"/csrData/citeseer_weights2.txt";

    #endif
#endif








// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                   TIME STAMP              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
double getTimestamp(){
	struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_usec + tv.tv_sec * 1e6;
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                        RELU             						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
static void relu(DTYPE *D){
  
    for(int i = 0; i < N_adj; i++){
        for(int j = 0; j < P_w; j++){
            
            if(D[i*P_w + j] < 0)
                D[i*P_w + j] = 0;
        }
    }
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		               LOAD WEIGHTS                						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
static void load_weights(int N, int M, BTYPE *A, std::string file_name){


    std::ifstream myFile(file_name);
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");


	std::string line;
	
	BTYPE val;
	int val_count=0;
	int val_zero=0;

    BTYPE array_val;


    for(int i = 0; i < N; i++){

    	std::getline(myFile, line);
    	std::stringstream ss(line);

        for(int j = 0; j < M; j++){

        	array_val = 0;
        	ss >> val;

        	if(val==0)
        		val_zero++;


        	array_val = (BTYPE)val;

	        if(ss.peek() == ',')
	        	ss.ignore();

	        A[i + j*N] = array_val;


	        val_count++;

	    }
    }

    cout << "**************************************************************************" << endl;
    cout << "Total " << sizeof(BTYPE)*8  << " bit values in weight matrix " << val_count << endl;
    cout << "Total values set to zero in weight matrix " << val_zero << endl;

    cout << "**************************************************************************" << endl;
}






// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                LOAD ADJACENCY               					//
// 																	    //
// ==================================================================== //
// ==================================================================== //
void loadcsr_adj( std::string file_name, 
                  int N, 
                  int M, 
                  
                  ATYPE *array_values,
                  
                  int *array_colIndices,
                  int *array_rowPtr,
                  int nnz_value){


	std::string line;
	std::ifstream inFile(file_name);

	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		cout << "reading " << file_name << " file" << endl;

    std::getline(inFile, line);

	std::stringstream ss;
	ss << line;

	std::string s = ss.str();

	for(int i = 0; i < N+1; i++){
		int temp;
		ss >> temp;

       array_rowPtr[i] = temp;

       if(ss.peek() == ',') ss.ignore();
	}


    std::getline(inFile, line);

	ss.str("");
    ss.clear();
	ss <<  line;


    for(int i = 0; i <  nnz_value; i++){
        ss >> array_colIndices[i];
        if(ss.peek() == ',') ss.ignore();
    }


    std::getline(inFile, line);

    ss.str("");
    ss.clear();
    ss << line;

    for (int i = 0; i <  nnz_value; i++) {
    	ATYPE float_val;
        ss >> float_val;

        // array_values[i] = scale_quantization(float_val, 2.12);
        // float threshold = 0.658;
        // if (float_val < -threshold)
        //     float_val = -threshold;
        // else if (float_val >= threshold)
        //     float_val = threshold;

        array_values[i] = (ATYPE) float_val;


        if(ss.peek() == ',') ss.ignore();
    }


    inFile.close();

    cout << "Number of non-zeros adj values in CSR file: " << nnz_value << endl;
	cout << "adj matrix size: " << M*N << endl;
	cout << "Total percentage of zero values in adj: " << (float)(M*N-nnz_value)/(float)(M*N) << endl;

	cout << "-----------------------------------------------------------------------------------\n" << endl;

}





// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                LOAD FEATURE              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
void loadcsr_fea( std::string file_name,
                  bool gemm_mode,
                  int N,
                  int M,

                  FTYPE *array_values,

                  int *array_colIndices,
                  int *array_rowPtr,
                  int nnz_value) {
	int i;

	std::string line;

	std::ifstream inFile(file_name);

	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		cout << "reading " << file_name << " file" << endl;


    std::getline(inFile, line);

	std::stringstream ss;
	ss << line;


    if(gemm_mode == 0){

        for (i = 0; i < N+1; i++) {
            int temp;
            ss >> temp;

            array_rowPtr[i] = temp;

            if(ss.peek() == ',') ss.ignore();
        }



        std::getline(inFile, line);

        ss.str("");
        ss.clear();
        ss <<  line;


        for (i = 0; i <  nnz_value; i++) {
            ss >> array_colIndices[i];

            if(ss.peek() == ',') ss.ignore();
        }

        std::getline(inFile, line);

        ss.str("");
        ss.clear();
        ss<<line;

    }

	
	for (i = 0; i <  nnz_value; i++) {
		FTYPE float_val;
		ss >> float_val;

		array_values[i] = (FTYPE)float_val;

	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();

    cout << "Number of non-zeros fea values in CSR file: " << nnz_value << endl;
	cout << "fea matrix size: " << M * N << endl;
	cout << "Total percentage of zero values in fea: " << (float)(M * N - nnz_value)/(float)(M * N) << endl;

	cout << "-----------------------------------------------------------------------------------\n" << endl;


}




// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                       MAIN                						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
int main(int argc, char* argv[]){

    int test_passed = 0;

    BTYPE *w_m, *w_m2;

    DTYPE *D2, *D;
    FTYPE *values_fea, *values_fea2;

    int *colIndices_fea, *rowPtr_fea;

    ATYPE *values_adj;
    int *colIndices_adj, *rowPtr_adj;

    int bias_count, nnz_fea, nnz_adj;

	
    cout << "\n-----------------------------------------------------------------------------------" << endl;
    cout << "Matrix dimensions N_adj / M_adj " << N_adj << ", M_fea: " << M_fea << ", P_w: " << P_w << endl;



    // --------------------------------------------------------------------------------
    //                      LAYER 1
    // --------------------------------------------------------------------------------
    values_fea = new FTYPE[max_M_fea * max_N_adj];
    colIndices_fea  = new int[max_M_fea * 100];
    rowPtr_fea  = new int[max_N_adj];

    values_adj  = new ATYPE[100 * max_N_adj];
    colIndices_adj  = new int[100 * max_N_adj];
    rowPtr_adj  = new int[max_N_adj];

    w_m = new BTYPE[max_M_fea * max_P_w];
    D = new DTYPE[N_adj * P_w];



    bool gemm_mode = 0;


    loadcsr_fea(fea_name, gemm_mode, N_adj, M_fea, values_fea, colIndices_fea, rowPtr_fea, NNZ_fea);
    loadcsr_adj(adj_name, N_adj, N_adj, values_adj, colIndices_adj, rowPtr_adj, NNZ_adj);
    load_weights(M_fea, P_w, w_m, w_name);

    double start_time, end_time, execution_time;


    start_time = getTimestamp();

    gfade(gemm_mode, N_adj, N_adj, M_fea, P_w, D, w_m, N_adj, rowPtr_fea, colIndices_fea, values_fea, rowPtr_adj,
       		  colIndices_adj, values_adj);
    int count = 0;
    for(int i = 0; i < N_adj; i++){
        for(int j = 0; j < P_w; j++){
//        	count++;
//        	cout << D[i*P_w + j] << endl;
//            std::ofstream outfile;
//            outfile.open("layer1_half.txt", std::ios_base::app);
//            outfile << D[i*P_w + j] << std::endl;
//            outfile.close();
        }
    }

//    cout << count << endl;



    // --------------------------------------------------------------------------------
    //                      LAYER 2
    // --------------------------------------------------------------------------------
     #ifdef layer2

        
         gemm_mode = 1;

         values_fea2 = new FTYPE[max_M_fea * max_N_adj];
         w_m2 = new BTYPE[max_M_fea * max_P_w];
         D2 = new DTYPE[N_adj2 * P_w2];


     	load_weights(M_fea2, P_w2, w_m2, w_name2);
        relu(D);

        
        for(int i = 0; i < N_adj; i++){
            for(int j = 0; j < P_w; j++){
                values_fea2[i*P_w + j] = (FTYPE)D[i*P_w + j];
            }
        }

//        values_fea2 = D;

        gfade(gemm_mode, N_adj2, N_adj2, M_fea2, P_w2, D2, w_m2, N_adj2, rowPtr_fea, colIndices_fea, values_fea2, rowPtr_adj,
                 colIndices_adj, values_adj);

        
         for(int i = 0; i < N_adj2; i++){
             for(int j = 0; j < P_w2; j++){
//            	 cout << D2[i*P_w2 + j] << endl;
                 std::ofstream outfile;
                 outfile.open("citeseer_test.txt", std::ios_base::app);
                 outfile << D2[i*P_w2 + j] << std::endl;
                 outfile.close();
             }
         }
    
	 #endif


    // --------------------------------------------------------------------------------
    //                      
    // --------------------------------------------------------------------------------
    end_time = getTimestamp();

    execution_time = (end_time - start_time) / (1000);

    cout << "CPU " << " Total execution time = " << execution_time << " msec" << endl;

    return 0;
}






// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                     csrDataS              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
#ifdef pubmed

    
    int N_adj = 19717;  // number of nodes
    int M_fea = 500;  // number of input features
    int P_w = 18;  // number of features in the hidden layer
    int NNZ_adj = 108365;  // number of non-zero values of adjacency
    int NNZ_fea = 988031;  // number of non-zero values of feature

    static const std::string adj_name = address +"/csrData/pubmed_adj.txt";
    static const std::string fea_name = address +"/csrData/pubmed_fea.txt";
    static const std::string w_name = address +"/csrData/pubmed_weights.txt";
    

    #ifdef layer2
        int N_adj2 = 19717;  // number of nodes
        int M_fea2 = 18;  // number of input features
        int P_w2 = 3;  // number of features in the hidden layer
        int NNZ_adj2 = 108365;  // number of non-zero values of adjacency
        int NNZ_fea2 = 19717 * 18;  // number of non-zero values of feature

        static const std::string adj_name2 = address +"/csrData/pubmed_adj.txt";
        static const std::string fea_name2 = address +"/csrData/pubmed_fea.txt";
        static const std::string w_name2 = address +"/csrData/pubmed_weights2.txt";
    #endif

  
#endif

#ifdef cora

    
    int N_adj = 2708;  // number of nodes
    int M_fea = 1433;  // number of input features
    int P_w = 64;  // number of features in the hidden layer
    int NNZ_adj = 13264;  // number of non-zero values of adjacency
    int NNZ_fea = 49216;  // number of non-zero values of feature

    static const std::string adj_name = address +"/csrData/cora_adj.txt";
    static const std::string fea_name = address +"/csrData/cora_fea.txt";
    static const std::string w_name = address +"/csrData/cora_weights.txt";
  

    #ifdef layer2
        int N_adj2 = 2708;  // number of nodes
        int M_fea2 = 64;  // number of input features
        int P_w2 = 7;  // number of features in the hidden layer
        int NNZ_adj2 = 13264;  // number of non-zero values of adjacency
        int NNZ_fea2 = 2708 * 64;  // number of non-zero values of feature

        static const std::string adj_name2 = address +"/csrData/cora_adj.txt";
        static const std::string fea_name2 = address +"/csrData/cora_fea.txt";
        static const std::string w_name2 = address +"/csrData/cora_weights2.txt";
    #endif
    
#endif

#ifdef citeseer
    
    int N_adj = 3327;  // number of nodes
    int M_fea = 3703;  // number of input features
    int P_w = 21;  // number of features in the hidden layer
    int NNZ_adj = 12431;  // number of non-zero values of adjacency
    int NNZ_fea = 105165;//105165;  // number of non-zero values of feature

    static const std::string adj_name = address + "/csrData/citeseer_adj.txt";
    static const std::string fea_name = address +"/csrData/citeseer_fea.txt";
    static const std::string w_name = address +"/csrData/citeseer_weights.txt";

   

    #ifdef layer2

        int N_adj2 = 3327;  // number of nodes
        int M_fea2 = 21; //3703;  // number of input features
        int P_w2 = 6; //21;  // number of features in the hidden layer
        int NNZ_adj2 = 12431;  // number of non-zero values of adjacency
        int NNZ_fea2 = 3327*21;//105165;  // number of non-zero values of feature

        static const std::string adj_name2 = address + "/csrData/citeseer_adj.txt";
//        static const std::string fea_name2 = address +"/csrData/citeseer_feat.txt";
        static const std::string w_name2 = address +"/csrData/citeseer_weights2.txt";

    #endif
#endif








// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                   TIME STAMP              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
double getTimestamp(){
	struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_usec + tv.tv_sec * 1e6;
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                        RELU             						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
static void relu(DTYPE *D){
  
    for(int i = 0; i < N_adj; i++){
        for(int j = 0; j < P_w; j++){
            
            if(D[i*P_w + j] < 0)
                D[i*P_w + j] = 0;
        }
    }
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		               LOAD WEIGHTS                						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
static void load_weights(int N, int M, BTYPE *A, std::string file_name){


    std::ifstream myFile(file_name);
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");


	std::string line;
	
	BTYPE val;
	int val_count=0;
	int val_zero=0;

    BTYPE array_val;


    for(int i = 0; i < N; i++){

    	std::getline(myFile, line);
    	std::stringstream ss(line);

        for(int j = 0; j < M; j++){

        	array_val = 0;
        	ss >> val;

        	if(val==0)
        		val_zero++;


        	array_val = (BTYPE)val;

	        if(ss.peek() == ',')
	        	ss.ignore();

	        A[i + j*N] = array_val;


	        val_count++;

	    }
    }

    cout << "**************************************************************************" << endl;
    cout << "Total " << sizeof(BTYPE)*8  << " bit values in weight matrix " << val_count << endl;
    cout << "Total values set to zero in weight matrix " << val_zero << endl;

    cout << "**************************************************************************" << endl;
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                LOAD ADJACENCY               					//
// 																	    //
// ==================================================================== //
// ==================================================================== //
void loadcsr_adj( std::string file_name, 
                  int N, 
                  int M, 
                  
                  ATYPE *array_values,
                  
                  int *array_colIndices,
                  int *array_rowPtr,
                  int nnz_value){


	std::string line;
	std::ifstream inFile(file_name);

	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		cout << "reading " << file_name << " file" << endl;

    std::getline(inFile, line);

	std::stringstream ss;
	ss << line;

	std::string s = ss.str();

	for(int i = 0; i < N+1; i++){
		int temp;
		ss >> temp;

       array_rowPtr[i] = temp;

       if(ss.peek() == ',') ss.ignore();
	}


    std::getline(inFile, line);

	ss.str("");
    ss.clear();
	ss <<  line;


    for(int i = 0; i <  nnz_value; i++){
        ss >> array_colIndices[i];
        if(ss.peek() == ',') ss.ignore();
    }


    std::getline(inFile, line);

    ss.str("");
    ss.clear();
    ss << line;

    for (int i = 0; i <  nnz_value; i++) {
    	ATYPE float_val;
        ss >> float_val;

        // array_values[i] = scale_quantization(float_val, 2.12);
        // float threshold = 0.658;
        // if (float_val < -threshold)
        //     float_val = -threshold;
        // else if (float_val >= threshold)
        //     float_val = threshold;

        array_values[i] = (ATYPE) float_val;


        if(ss.peek() == ',') ss.ignore();
    }


    inFile.close();

    cout << "Number of non-zeros adj values in CSR file: " << nnz_value << endl;
	cout << "adj matrix size: " << M*N << endl;
	cout << "Total percentage of zero values in adj: " << (float)(M*N-nnz_value)/(float)(M*N) << endl;

	cout << "-----------------------------------------------------------------------------------\n" << endl;

}





// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                LOAD FEATURE              						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
void loadcsr_fea( std::string file_name,
                  bool gemm_mode,
                  int N,
                  int M,

                  FTYPE *array_values,

                  int *array_colIndices,
                  int *array_rowPtr,
                  int nnz_value) {
	int i;

	std::string line;

	std::ifstream inFile(file_name);

	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		cout << "reading " << file_name << " file" << endl;


    std::getline(inFile, line);

	std::stringstream ss;
	ss << line;


    if(gemm_mode == 0){

        for (i = 0; i < N+1; i++) {
            int temp;
            ss >> temp;

            array_rowPtr[i] = temp;

            if(ss.peek() == ',') ss.ignore();
        }



        std::getline(inFile, line);

        ss.str("");
        ss.clear();
        ss <<  line;


        for (i = 0; i <  nnz_value; i++) {
            ss >> array_colIndices[i];

            if(ss.peek() == ',') ss.ignore();
        }

        std::getline(inFile, line);

        ss.str("");
        ss.clear();
        ss<<line;

    }

	
	for (i = 0; i <  nnz_value; i++) {
		FTYPE float_val;
		ss >> float_val;

		array_values[i] = (FTYPE)float_val;

	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();

    cout << "Number of non-zeros fea values in CSR file: " << nnz_value << endl;
	cout << "fea matrix size: " << M * N << endl;
	cout << "Total percentage of zero values in fea: " << (float)(M * N - nnz_value)/(float)(M * N) << endl;

	cout << "-----------------------------------------------------------------------------------\n" << endl;


}




// ==================================================================== //
// ==================================================================== //
// 																	    //
// 		                       MAIN                						//
// 																	    //
// ==================================================================== //
// ==================================================================== //
int main(int argc, char* argv[]){

    int test_passed = 0;

    BTYPE *w_m, *w_m2;

    DTYPE *D2, *D;
    FTYPE *values_fea, *values_fea2;

    int *colIndices_fea, *rowPtr_fea;

    ATYPE *values_adj;
    int *colIndices_adj, *rowPtr_adj;

    int bias_count, nnz_fea, nnz_adj;

	
    cout << "\n-----------------------------------------------------------------------------------" << endl;
    cout << "Matrix dimensions N_adj / M_adj " << N_adj << ", M_fea: " << M_fea << ", P_w: " << P_w << endl;



    // --------------------------------------------------------------------------------
    //                      LAYER 1
    // --------------------------------------------------------------------------------
    values_fea = new FTYPE[max_M_fea * max_N_adj];
    colIndices_fea  = new int[max_M_fea * 100];
    rowPtr_fea  = new int[max_N_adj];

    values_adj  = new ATYPE[100 * max_N_adj];
    colIndices_adj  = new int[100 * max_N_adj];
    rowPtr_adj  = new int[max_N_adj];

    w_m = new BTYPE[max_M_fea * max_P_w];
    D = new DTYPE[N_adj * P_w];



    bool gemm_mode = 0;


    loadcsr_fea(fea_name, gemm_mode, N_adj, M_fea, values_fea, colIndices_fea, rowPtr_fea, NNZ_fea);
    loadcsr_adj(adj_name, N_adj, N_adj, values_adj, colIndices_adj, rowPtr_adj, NNZ_adj);
    load_weights(M_fea, P_w, w_m, w_name);

    double start_time, end_time, execution_time;


    start_time = getTimestamp();

    gfades(gemm_mode, N_adj, N_adj, M_fea, P_w, w_m, D, D, D, D, N_adj,
             rowPtr_fea, rowPtr_fea, rowPtr_fea, rowPtr_fea,
             colIndices_fea, colIndices_fea, colIndices_fea, colIndices_fea,
             values_fea, values_fea, values_fea, values_fea,
             rowPtr_adj, rowPtr_adj, rowPtr_adj, rowPtr_adj,
             colIndices_adj, colIndices_adj, colIndices_adj, colIndices_adj,
             values_adj, values_adj, values_adj, values_adj);

    int count = 0;
    for(int i = 0; i < N_adj; i++){
        for(int j = 0; j < P_w; j++){
//        	count++;
//        	cout << D[i*P_w + j] << endl;
//            std::ofstream outfile;
//            outfile.open("layer1_half.txt", std::ios_base::app);
//            outfile << D[i*P_w + j] << std::endl;
//            outfile.close();
        }
    }

//    cout << count << endl;



    // --------------------------------------------------------------------------------
    //                      LAYER 2
    // --------------------------------------------------------------------------------
     #ifdef layer2

        
         gemm_mode = 1;

         values_fea2 = new FTYPE[max_M_fea * max_N_adj];
         w_m2 = new BTYPE[max_M_fea * max_P_w];
         D2 = new DTYPE[N_adj2 * P_w2];


     	load_weights(M_fea2, P_w2, w_m2, w_name2);
        relu(D);

        
        for(int i = 0; i < N_adj; i++){
            for(int j = 0; j < P_w; j++){
                values_fea2[i*P_w + j] = (FTYPE)D[i*P_w + j];
            }
        }

//        values_fea2 = D;


        gfades(gemm_mode, N_adj2, N_adj2, M_fea2, P_w2, w_m2, D2, D2, D2, D2, N_adj2,
             rowPtr_fea, rowPtr_fea, rowPtr_fea, rowPtr_fea,
             colIndices_fea, colIndices_fea, colIndices_fea, colIndices_fea,
             values_fea2, values_fea2, values_fea2, values_fea2,
             rowPtr_adj, rowPtr_adj, rowPtr_adj, rowPtr_adj,
             colIndices_adj, colIndices_adj, colIndices_adj, colIndices_adj,
             values_adj, values_adj, values_adj, values_adj);

        
         for(int i = 0; i < N_adj2; i++){
             for(int j = 0; j < P_w2; j++){
//            	 cout << D2[i*P_w2 + j] << endl;
                 std::ofstream outfile;
                 outfile.open("citeseer_test.txt", std::ios_base::app);
                 outfile << D2[i*P_w2 + j] << std::endl;
                 outfile.close();
             }
         }
    
	 #endif


    // --------------------------------------------------------------------------------
    //                      
    // --------------------------------------------------------------------------------
    end_time = getTimestamp();

    execution_time = (end_time - start_time) / (1000);

    cout << "CPU " << " Total execution time = " << execution_time << " msec" << endl;

    return 0;
}

