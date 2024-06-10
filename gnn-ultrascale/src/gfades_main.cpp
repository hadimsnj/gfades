// ==================================================================== //
// ==================================================================== //
// 	This file is part of the gFADES GNN accelerator has been written    //
//  at Linkoping University for the WASP project						//
// 						        								        //
// 	Author : Jose Nunez-Yanez											//
// ==================================================================== //
// ==================================================================== //

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream> 
#include <hls_stream.h>

#include "ap_int.h"
#include "gfades.h"
#include "hls_streamofblocks.h"

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						        								        //
// 																	    //
// ==================================================================== //
// ==================================================================== //

typedef ITYPE buf[B_HEIGHT/FEA_THREADS][C_WIDTH_BLOCK];

const int BLOCK=B_WIDTH_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
const int SBLOCK=SPMM_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK

const int PARALLEL_ROW = B_BLOCK_PARALLEL;
const int A_WIDTH_FIFO =  A_WIDTH;
const int FIFO_DEPTH = MAX_FIFO;

const int FADD_LATENCY_ADJ = FTYPE_LATENCY_ADJ;
const int FADD_LATENCY_FEA = FTYPE_LATENCY_FEA;

#ifdef simulation
    extern float max_adj;
    extern float min_adj;
    extern float max_fea;
    extern float min_fea;
    extern float acc2_fea_min;
    extern float acc2_fea_max;
    extern float acc2_adj_min;
    extern float acc2_adj_max;
#endif



// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL FLOAT ADJ 1 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_float_adj_1(ATYPE a_value,
                            ITYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],
                            
                            INT32 b_row,
                            
                            DTYPE acc[B_WIDTH_BLOCK])
{
	
	#pragma HLS INLINE

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
			
        ITYPE b_val = b_block[b_row][j];
        ATYPE a_val = a_value;

        acc[j] = (DTYPE)a_val*(DTYPE)b_val;
	} //j loop


}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL FLOAT ADJ 2 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_float_adj_2(int block_size,
                            ATYPE a_value,
                            
                            BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
                            BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],
                            
                            INT32 b_row,
                            
                            ITYPE acc[B_WIDTH_BLOCK])
{

	#pragma HLS INLINE

	for(int j = 0; j < B_WIDTH_BLOCK; j++){

        ATYPE a_val = a_value;
        BTYPE b_val;


        int sel_block; // = (b_row>>(log2N-2))&0x3;
        int b_row_block;

        

        if (b_row < block_size){
            b_row_block = b_row;
            sel_block = 0;
        }
        if (b_row > (block_size-1)){
            b_row_block = b_row-block_size;
            sel_block = 1;
        }



        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];

        switch(sel_block){
            case 0:
                b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                break;
            case 1:
                b_val = b_val2;
            break;
        }

        acc[j] = (ITYPE)a_val * (ITYPE)b_val;
	} //j loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL FLOAT ADJ 4 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_float_adj_4(int block_size,
                            ATYPE a_value,
                            BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
                            BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],
                            BTYPE b_block3[B_HEIGHT][B_WIDTH_BLOCK],
                            BTYPE b_block4[B_HEIGHT][B_WIDTH_BLOCK],
                            INT32 b_row,
                            
                            ITYPE acc[B_WIDTH_BLOCK])
{
	
	#pragma HLS INLINE

	for(int j = 0; j < B_WIDTH_BLOCK; j++){

	  		ATYPE a_val = a_value;
	  		BTYPE b_val;

	  		int sel_block; 
	  		int b_row_block;

            if (b_row < block_size){
	  			b_row_block = b_row;
	  			sel_block = 0;
	  		}
	  		
            
            if (b_row > (block_size-1)){
	  			b_row_block = b_row-block_size;
	  			sel_block = 1;
	  		}


  		    if (b_row > (2*block_size-1) && b_row < 3*block_size){
  		  	    b_row_block = b_row-2*block_size;
	  			sel_block = 2;
	  		}

  		    if (b_row > 3*block_size-1){
  			    b_row_block = b_row-3*block_size;
	  			sel_block = 3;
	  		}

	  		BTYPE b_val1 = b_block1[b_row_block][j];
			BTYPE b_val2 = b_block2[b_row_block][j];
			BTYPE b_val3 = b_block3[b_row_block][j];
			BTYPE b_val4 = b_block4[b_row_block][j];


	  		switch(sel_block){
	  			case 0:
	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
	  				break;
	  			case 1:
	  				b_val = b_val2;
  				break;
	  			case 2:
	  				b_val = b_val3;
  				break;
	  			case 3:
	  				b_val = b_val4;
  				break;
	  		}

			acc[j] = (ITYPE)a_val*(ITYPE)b_val;
	} // j loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL FLOAT FEA 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_float_fea(ATYPE a_value,
                        BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],
                        INT32 b_row,
                        
                        ITYPE acc[B_WIDTH_BLOCK])
{
	#pragma HLS INLINE

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
	
        BTYPE b_val = b_block[b_row][j];
        ATYPE a_val = a_value;

        acc[j] = (ITYPE)a_val*(ITYPE)b_val;


	} // j loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL INT ADJ 1 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_int_adj_1(int block_size,
                        ATYPE a_value,
                        ITYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],

                        INT32 b_row,
                        
                        ITYPE acc[B_WIDTH_BLOCK])
{


	for(int j = 0; j < B_WIDTH_BLOCK; j++){
        #pragma HLS UNROLL

        acc[j] = 0;
    }

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
	
        ATYPE a_val = a_value;
        BTYPE b_val;

        int sel_block; 
        int b_row_block;

        if (b_row < block_size){
            b_row_block = b_row;
            sel_block = 0;
        }
        
        BTYPE b_val1 = b_block1[b_row_block][j];
    
        switch(sel_block){
            case 0:
                b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                break;
        }

        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        ITYPE acc_i = a_val_i*b_val_i;
        acc[j] += acc_i;
			
	} //j loop

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL INT ADJ 2 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_int_adj_2(int block_size,
                        ATYPE a_value,
                        
                        ITYPE b_block1[B_HEIGHT/2][B_WIDTH_BLOCK],
                        ITYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
                        
                        INT32 b_row,
                        
                        ITYPE acc[B_WIDTH_BLOCK])
{


	for(int j = 0; j < B_WIDTH_BLOCK; j++){
		#pragma HLS UNROLL

		acc[j] = 0;
    }

	for(int j = 0; j < B_WIDTH_BLOCK; j++){

        ATYPE a_val = a_value;
        BTYPE b_val;

        int sel_block; // = (b_row>>(log2N-2))&0x3;
        int b_row_block;

        if (b_row < block_size){
            b_row_block = b_row;
            sel_block = 0;
        }
        
        if (b_row > (block_size-1)){
            b_row_block = b_row-block_size;
            sel_block = 1;
        }

        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];

        switch(sel_block){
            case 0:
                b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                break;
            case 1:
                b_val = b_val2;
                break;
        }
        
        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        ITYPE acc_i = a_val_i*b_val_i;
        acc[j] += acc_i;
			
	} // j loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				       DSP KERNEL INT ADJ 4 						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_int_adj_4(int block_size,
                        ATYPE a_value,
                        ITYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],

                        ITYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
                        ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],

                        ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
                        INT32 b_row,
                        
                        ITYPE acc[B_WIDTH_BLOCK])
{


	for(int j = 0; j < B_WIDTH_BLOCK; j++){
			#pragma HLS UNROLL

			acc[j] = 0;
    }

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
		
        ATYPE a_val = a_value;
        BTYPE b_val;


        int sel_block; // = (b_row>>(log2N-2))&0x3;
        int b_row_block;

        
        if (b_row < block_size){
            b_row_block = b_row;
            sel_block = 0;
        }

        if (b_row > (block_size-1) && b_row < 2*block_size){
            b_row_block = b_row-block_size;
            sel_block = 1;
        }

        if (b_row > (2*block_size-1) && b_row < 3*block_size){
            b_row_block = b_row-2*block_size;
            sel_block = 2;
        }

        if (b_row > 3*block_size-1){
            b_row_block = b_row-3*block_size;
            sel_block = 3;
        }
       
        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];
        BTYPE b_val3 = b_block3[b_row_block][j];
        BTYPE b_val4 = b_block4[b_row_block][j];


        switch(sel_block){
            case 0:
                b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                break;
            case 1:
                b_val = b_val2;
                break;
            case 2:
                b_val = b_val3;
                break;
            case 3:
                b_val = b_val4;
                break;
        }

       
        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        ITYPE acc_i = a_val_i * b_val_i;
        acc[j] += acc_i;

	} // j loop
}
// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				            DSP KERNEL INT FEA						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_int_fea(FTYPE a_value, 
                        BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK], 

                        INT32 b_row, 
                        
                        ITYPE acc[B_WIDTH_BLOCK])
{

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
        #pragma HLS UNROLL

        acc[j] = 0;
    }

	for(int j = 0; j < B_WIDTH_BLOCK; j++){
		
        FTYPE a_val = a_value;
        BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.
        
        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        ITYPE acc_i = a_val_i*b_val_i;
        acc[j] += acc_i;
			
	} //j loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				            WRITEC            						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void writec(bool relu, 
            int first_row, 
            int row_count, 
            int P, 
            hls::stream<ITYPE> write_fifo[C_WIDTH_BLOCK][SPMM_BLOCK], 
            DTYPE* C, 
            int B_index, 
            int B_index_loop, 
            int tail)
{
    int B_WIDTH_INT;
    int WL;

    #if defined FLOAT
        WL = row_count;
    #endif

    #if defined HALF
        WL = row_count;
    #endif

    #ifdef EIGHTBIT
        WL = row_count;
    #endif



    if (B_index < (B_index_loop-1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;


		

	#if (USE_SBLOCKS == 1)
        
        LOOP_WRITE1:for(int i = 0; i < WL; i+=SPMM_BLOCK){
            DTYPE C_out;
            
            LOOP_WRITE3:for(int z = 0; z <  SPMM_BLOCK; z++){
                
                if ((z+i) < WL){
                    #if (USE_TAIL == 1)
                        LOOP_WRITE2: for(int j = 0; j <  B_WIDTH_INT; j++) 
                    #else
                        LOOP_WRITE2: for(int j = 0; j <  B_WIDTH_BLOCK; j++)
                    #endif
                        {
                            #pragma HLS PIPELINE
                            
                            C_out =  write_fifo[j][z].read();
                                
                            #if (USE_RELU == 1)
                                if(j < B_WIDTH_INT){
                                    C[(i+z)*P+j+B_index*B_WIDTH_BLOCK] = C_out;
                                }
                            #else
                                    C[(i+z)*P+j+B_index*B_WIDTH_BLOCK] = C_out;
                            #endif
                               
                                
                        }
                }
            }
        }
    #endif

    #if (USE_SBLOCKS == 0)
		LOOP_WRITE4:for(int i = 0; i < WL; i++){
			DTYPE C_out;
            
            #if (USE_TAIL == 1)
                LOOP_WRITE5:for(int j = 0; j <  B_WIDTH_INT; j++) //this reduces performance
            #else
                LOOP_WRITE5:for(int j = 0; j <  B_WIDTH_BLOCK; j++) 
            #endif
                {                                                                    
                    #pragma HLS PIPELINE
                    
                    C_out =  write_fifo[j][0].read();
                    #if (USE_RELU == 1)
                        if(j<B_WIDTH_INT){
                            if (C_out > 0 || relu == 0)
                                C[i*P+j+B_index*B_WIDTH_BLOCK] = C_out;
                            else
                                C[i*P+j+B_index*B_WIDTH_BLOCK] = 0.0;
                        }
                    #else
                        C[i*P+j+B_index*B_WIDTH_BLOCK] = C_out;
                    #endif
                }
        }
    #endif
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READPTR_FEA        						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void readptr_fea(bool gemm_mode, 
                int N, 
                int M, 
                int *rowPtr, 
                hls::stream<int> rnnz_fifo[SPMM_BLOCK])
{

    #pragma HLS inline off
	int rnnz, current_index, next_index;


	if (gemm_mode==0){
		current_index= rowPtr[0];
		
		LOOP_A_INDEX_SPMM1 : for(int A_index = 0; A_index < N; A_index+=SPMM_BLOCK){
			int brnnz = 0;
			
            LOOP_B_INDEX_SPMM1 : for(int B_index = 0; B_index < SPMM_BLOCK; B_index++){
			    #pragma HLS PIPELINE
				
                if((A_index+B_index) < N){
					next_index=rowPtr[A_index+B_index+1];
					rnnz = next_index-current_index;
					brnnz+=rnnz;
					current_index = next_index;
					
                    rnnz_fifo[B_index] << brnnz;
					
				}else{
					rnnz_fifo[B_index] << brnnz; //juse use the last value of rnnz in block
				}
			}
        }
	
    }else{
		
        LOOP_A_INDEX_SPMM2 : for(int A_index = 0; A_index < N; A_index+=SPMM_BLOCK){
			int brnnz = 0;
			
            LOOP_B_INDEX_SPMM2 : for(int B_index = 0; B_index < SPMM_BLOCK; B_index++){
			    #pragma HLS PIPELINE
                
                if((A_index+B_index) < N){
                    rnnz = M;
                    brnnz+=rnnz;
                    rnnz_fifo[B_index] << brnnz;
                }else{
                    rnnz_fifo[B_index] << brnnz; //juse use the last value of rnnz in block
                }
		    }
        }
	}//end else
}



// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READPTR_ADJ        						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void readptr_adj(int N, 
                 int *rowPtr, 
                 hls::stream<int> rnnz_fifo[SPMM_BLOCK])
{
	#pragma HLS inline off
	
    int rnnz, current_index, next_index;
	current_index= rowPtr[0];

    LOOP_A_INDEX_SPMM1 : for(int A_index = 0; A_index < N; A_index+=SPMM_BLOCK){
        int brnnz=0;
        
        LOOP_B_INDEX_SPMM1 : for(int B_index = 0; B_index < SPMM_BLOCK; B_index++){
            #pragma HLS PIPELINE
            
            if((A_index + B_index) < N){
                next_index = rowPtr[A_index+B_index+1];
                rnnz = next_index-current_index;
                brnnz += rnnz;
                current_index = next_index;
                
                rnnz_fifo[B_index] << brnnz;
            
            }else
                rnnz_fifo[B_index] << brnnz; //juse use the last value of rnnz in block
        }
    }
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READAVAL_ADJ        						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void readval_adj(int last_index, 
                
                hls::stream<ATYPE> &A_fifo, 
                hls::stream<int> &col_indices_fifo, 
                
                ATYPE *values, 
                int *columnIndex)
{
		#pragma HLS inline off
		
        LOOP_J_SPMM:for(int j = 0; j < last_index; j++){
			#pragma HLS PIPELINE
			
            A_fifo <<  values[j];
			col_indices_fifo << columnIndex[j];
		}

}

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READAVAL_FEA        						    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void readval_fea(bool gemm_mode, 
                int ccount, 
                int last_index, 
                
                hls::stream<FTYPE> &A_fifo, 
                hls::stream<int> &col_indices_fifo, 
                
                FTYPE *values, 
                int *columnIndex)
{
	#pragma HLS inline off
	
    if (gemm_mode==0){
		
        LOOP_J_SPMM1:for(int j = 0; j < last_index; j++){
			#pragma HLS PIPELINE

			A_fifo <<  values[j];

			col_indices_fifo << columnIndex[j];
		}
	
    }else{
        int c=0;
        
        LOOP_J_SPMM2:for(int j = 0; j < last_index; j++){
            #pragma HLS PIPELINE
    
            A_fifo <<  values[j];
            col_indices_fifo << c;

            if (c == (ccount-1)) 
                c=0;
            else
                c++;
        }
	}
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READA1        						        //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void reada1(bool gemm_mode, 
            int M, 
            int first_row, 
            int row_count, 
            hls::stream<FTYPE> &A_fifo_fea, 
            hls::stream<int> &col_indices_fifo_fea, 
            hls::stream<int> rnnz_fifo_fea[SPMM_BLOCK], 

            int B_index_loop, 
            int tail, 
            int *rowPtr_fea, 
            int *columnIndex_fea, 
            FTYPE *values_fea)
{

	int last_index_fea;

	if (gemm_mode==0){
		last_index_fea=rowPtr_fea[first_row+row_count]-rowPtr_fea[first_row];
		columnIndex_fea += rowPtr_fea[first_row];
		values_fea += rowPtr_fea[first_row];
	    rowPtr_fea += first_row;
	
    }else{
		last_index_fea=row_count*M;
		values_fea+=first_row*M;
	}

	readptr_fea(gemm_mode, row_count, M, rowPtr_fea, rnnz_fifo_fea);
	readval_fea(gemm_mode, M, last_index_fea, A_fifo_fea, col_indices_fifo_fea, values_fea, columnIndex_fea);

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				          READA2        						        //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void reada2(int first_row, 
            int row_count, 
            int B_index_loop, 
            int tail, 
            
            hls::stream<ATYPE> &A_fifo_adj, 
            hls::stream<int> &col_indices_fifo_adj, 
            hls::stream<int> rnnz_fifo_adj[SPMM_BLOCK], 

            int *rowPtr_adj, 
            int *columnIndex_adj, 
            ATYPE *values_adj)
{

	int last_index_adj;

	last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];
	
	columnIndex_adj += rowPtr_adj[first_row];
	values_adj += rowPtr_adj[first_row];
	rowPtr_adj += first_row;
	
	readptr_adj(row_count, rowPtr_adj, rnnz_fifo_adj);
	readval_adj(last_index_adj, A_fifo_adj, col_indices_fifo_adj, values_adj, columnIndex_adj);

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				DSP KERNEL WRAPPER ADJ 4						        //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_wrapper_adj_4(  int block_size,
                                int M[SPMM_BLOCK],
                                hls::stream<ATYPE> &A_fifo,
                                hls::stream<int> &col_indices_fifo,
                                ITYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],

                                ITYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
                                ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
                                ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],

                                
                                
                                ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{

    #if defined FLOAT || defined HALF


        FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
        FTYPE acc_float[B_WIDTH_BLOCK];
        
        #pragma HLS ARRAY_PARTITION variable=acc_part complete
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL

            acc_float[j] = 0;
        }

        RESET_ACC_LOOP_SPMM: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 0; l < FADD_LATENCY_ADJ; l++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    #pragma HLS UNROLL
                    
                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];
        int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
        M_aux[0] = 0;
        
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            
            M_aux[j] = M[j-1];
        }
        

        DSP_LOOP_SPMM:for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ){
            #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

            DSP_LOOP_SPMM2:for(int i = 0; i < FADD_LATENCY_ADJ; i++){

                DTYPE v;
                int ci;
                
                if ((k+i) < BM){ //avoid trying to read empty FIFO that only contains M elements
                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                
                }else{
                    v=0;
                    ci=0;
                }

                dsp_kernel_float_adj_4(block_size, v, b_block1, b_block2, b_block3, b_block4, ci,   acc_float);


                for(int j = 0; j < B_WIDTH_BLOCK; j++){
                 
                    for(int z = 0; z < SPMM_BLOCK; z++){
                        #pragma HLS UNROLL
                            
                        if((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                            acc_part[i][j][z] += acc_float[j];
                    }
                    
                    #ifdef simulation
                        if (acc_part[i][j] > max_adj)
                            max_adj = acc_part[i][j];
                            
                        if (acc_part[i][j] < min_adj)
                            min_adj = acc_part[i][j];
                    #endif
                }

            } //i loop

        } //k loop

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 1; l < FADD_LATENCY_ADJ; l++){
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    
                    acc_part[0][j][z] += acc_part[l][j][z];
                }
            }
        }

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int z = 0; z < SPMM_BLOCK; z++){
                #pragma HLS UNROLL
                
                FTYPE acc_part_float = acc_part[0][j][z];
                acc2[j][z] = acc_part_float;
            }
        }
    #endif

	#ifdef EIGHTBIT

		ITYPE acc[B_WIDTH_BLOCK];
	    #pragma HLS ARRAY_PARTITION variable=acc complete

	
        int BM = M[SPMM_BLOCK-1];
        int M_aux[SPMM_BLOCK+1];
        M_aux[0] = 0;
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            M_aux[j] = M[j-1];
        }

	    DSP_LOOP_SPMM:for(int i = 0; i < BM; i+=1){
		 	#pragma HLS PIPELINE
        	 
			DTYPE v = A_fifo.read();
			int ci = col_indices_fifo.read();

			dsp_kernel_int_adj_4(block_size, v, b_block1, b_block2, b_block3, b_block4, ci,   acc);

			for(int j = 0; j < B_WIDTH_BLOCK; j++){
				#pragma HLS UNROLL

				for(int z = 0; z < SPMM_BLOCK; z++){
					#pragma HLS UNROLL
						
                        if (i >= M_aux[z] && i < M_aux[z + 1])
						    acc2[j][z] += acc[j];
				}//z loop
						
			}//j loop

	    } //i loop
	#endif

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				DSP KERNEL WRAPPER ADJ 2						        //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_wrapper_adj_2(int block_size,
                            int M[SPMM_BLOCK],
                            
                            hls::stream<ATYPE> &A_fifo,
                            hls::stream<int> &col_indices_fifo,

                            ITYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
                            ITYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
                            ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],

                            
                            
                            ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{



    #if defined FLOAT || defined HALF


        FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
        FTYPE acc_float[B_WIDTH_BLOCK];

        #pragma HLS ARRAY_PARTITION variable=acc_part complete
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL

            acc_float[j] = 0;
        }


        RESET_ACC_LOOP_SPMM: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 0; l < FADD_LATENCY_ADJ; l++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];
        int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
        M_aux[0] = 0;
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            
            M_aux[j] = M[j-1];
        }
    


        DSP_LOOP_SPMM:for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ){
            #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

            DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++){

                DTYPE v;
                int ci;
                
                if((k+i) < BM){ //avoid trying to read empty FIFO that only contains M elements
                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                }else{
                    v = 0;
                    ci = 0;
                }

                dsp_kernel_float_adj_2(block_size, v, b_block1, b_block2, ci,   acc_float);


                for(int j = 0; j < B_WIDTH_BLOCK; j++){
                    
                    for(int z = 0; z < SPMM_BLOCK; z++){
                        #pragma HLS UNROLL
                        
                        if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                            acc_part[i][j][z] += acc_float[j];
                    }

                    #ifdef simulation
                        if (acc_part[i][j] > max_adj)
                            max_adj = acc_part[i][j];
                        if (acc_part[i][j] < min_adj)
                            min_adj = acc_part[i][j];
                    #endif
                }
            }//i loop
        }// k loop

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 1; l < FADD_LATENCY_ADJ; l++){
                #pragma HLS unroll
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    acc_part[0][j][z] += acc_part[l][j][z];
                }
            }
        }

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int z = 0; z < SPMM_BLOCK; z++){
                FTYPE acc_part_float = acc_part[0][j][z];
                acc2[j][z] = acc_part_float;
            }
        }
    #endif

    #ifdef EIGHTBIT

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        //for(int j = 0; j < B_WIDTH_BLOCK; j++){

        //	#pragma HLS UNROLL

        //		acc2[j] = 0;
        //}


        int BM = M[SPMM_BLOCK-1];

        int M_aux[SPMM_BLOCK+1];
        M_aux[0] = 0;
        for(int j = 1; j < SPMM_BLOCK+1; j++)
        {
        #pragma HLS UNROLL
        M_aux[j] = M[j-1];
        }

        DSP_LOOP_SPMM: for(int i = 0; i < BM; i+=1){
        #pragma HLS PIPELINE
        //#pragma HLS UNROLL factor=PARALLEL_ROW
        DTYPE v = A_fifo.read();

        int ci = col_indices_fifo.read();

        dsp_kernel_int_adj_2(block_size,v,b_block1,b_block2,
        //b_block3,b_block4,
        ci,acc);



        for(int j = 0; j < B_WIDTH_BLOCK; j++){

        #pragma HLS UNROLL
        for(int z = 0; z < SPMM_BLOCK; z++)
        {
        #pragma HLS UNROLL
        if (i>=M_aux[z]&&i<M_aux[z+1])
        acc2[j][z] += acc[j];
        }//z loop
        //////std::cout << " compute2 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;
        }//j loop

        //////////std::cout << " compute1 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;


        } //i loop





    #endif

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 				DSP KERNEL WRAPPER ADJ 1						        //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_wrapper_adj_1(int block_size,
                            int M[SPMM_BLOCK],
                            
                            hls::stream<ATYPE> &A_fifo,
                            hls::stream<int> &col_indices_fifo,

                            ITYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
                            ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],

                            
                            
                            ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{



	#if defined FLOAT || defined HALF


        FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
        FTYPE acc_float[B_WIDTH_BLOCK];

        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL

            acc_float[j] = 0;
        }





        RESET_ACC_LOOP_SPMM: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 0; l < FADD_LATENCY_ADJ; l++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];
        int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
        M_aux[0] = 0;
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            
            M_aux[j] = M[j-1];
        }
       

        DSP_LOOP_SPMM:for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ){
            #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

            DSP_LOOP_SPMM2:for(int i = 0; i < FADD_LATENCY_ADJ; i++){

                DTYPE v;
                int ci;
                
                if ((k+i) < BM){ //avoid trying to read empty FIFO that only contains M elements
                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                
                }else{
                    v=0;
                    ci=0;
                }

                dsp_kernel_float_adj_1(v,b_block1,ci,acc_float);

                for(int j = 0; j < B_WIDTH_BLOCK; j++){
                
                    for(int z = 0; z < SPMM_BLOCK; z++){
                        #pragma HLS UNROLL
                        
                        if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                            acc_part[i][j][z] += acc_float[j];
                    }
                    
                    #ifdef simulation
                        if (acc_part[i][j] > max_adj)
                            max_adj = acc_part[i][j];
                        if (acc_part[i][j] < min_adj)
                            min_adj = acc_part[i][j];
                    #endif
                }
            }//i loop
        }//k loop

        ACC_PART1 : for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            ACC_PART2 : for(int z = 0; z < SPMM_BLOCK; z++){
                #pragma HLS UNROLL
                
                ACC_PART3 : for(int l = 1; l < FADD_LATENCY_ADJ; l++){
                    #pragma HLS PIPELINE=1
                    
                    acc_part[0][j][z] += acc_part[l][j][z];
                }
            }
        }

        FLOAT_PART1 : for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            FLOAT_PART2 : for(int z = 0; z < SPMM_BLOCK; z++){
                #pragma HLS UNROLL
                
                FTYPE acc_part_float = acc_part[0][j][z];
                acc2[j][z] = acc_part_float;
            }
        }
	#endif

    #ifdef EIGHTBIT

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete



        int BM = M[SPMM_BLOCK-1];
        int M_aux[SPMM_BLOCK+1];
        M_aux[0] = 0;

        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            M_aux[j] = M[j-1];
        }

        DSP_LOOP_SPMM: for(int i = 0; i < BM; i+=1){
            #pragma HLS PIPELINE
        
            DTYPE v = A_fifo.read();
            int ci = col_indices_fifo.read();

        
            dsp_kernel_int_adj_1(block_size, v, b_block1, ci,   acc);



            for(int j = 0; j < B_WIDTH_BLOCK; j++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    #pragma HLS UNROLL
                    
                    if (i>=M_aux[z]&&i<M_aux[z+1])
                        acc2[j][z] += acc[j];
                }//z loop
            }//j loop
        } //i loop
    #endif

}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 					DSP KERNEL WRAPPER FEA		    				    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void dsp_kernel_wrapper_fea(bool gemm_mode,
                            int M[SPMM_BLOCK],
                            
                            hls::stream<FTYPE> &A_fifo,
                            hls::stream<int> &col_indices_fifo,
                            
                            BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],
                            
                            
                            ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])

{

    #if defined FLOAT || defined HALF

        ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
        ITYPE acc_float[B_WIDTH_BLOCK];

        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL

            acc_float[j] = 0;
        }

        RESET_ACC_LOOP_SPMM:for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            for(int l = 0; l < FADD_LATENCY_FEA; l++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    #pragma HLS UNROLL
                    
                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];

        int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
        M_aux[0] = 0;
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            
            M_aux[j] = M[j-1];
        }
        

        DSP_LOOP_SPMM:for(int k = 0; k < BM; k+=FADD_LATENCY_FEA){
            #pragma HLS PIPELINE II=FADD_LATENCY_FEA

            DSP_LOOP_SPMM2:for(int i = 0; i < FADD_LATENCY_FEA; i++){
                DTYPE v;
                int ci;
                
                if ((k+i) < BM){ //avoid trying to read empty FIFO that only contains BM elements
                
                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                    
                }else{
                    
                    v=0;
                    ci=0;
                }

                dsp_kernel_float_fea(v,b_block,ci,acc_float);

                SPMM_BLOCK_LOOP1 :for(int j = 0; j < B_WIDTH_BLOCK; j++){
                    #pragma HLS UNROLL
                    
                    SPMM_BLOCK_LOOP2 : for(int z = 0; z < SPMM_BLOCK; z++){
                        #pragma HLS PIPELINE II=1
                        
                        if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                                acc_part[i][j][z] += acc_float[j];
                    }//z loop
                } //j loop
            } //i loop
        } // k loop

        ACC_PART1 : for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            ACC_PART2: for(int z = 0; z < SPMM_BLOCK; z++){
                #pragma HLS UNROLL
                
                ACC_PART3 : for(int l = 1; l < FADD_LATENCY_FEA; l++){
                    #pragma HLS PIPELINE II=1
                    
                    acc_part[0][j][z] += acc_part[l][j][z];
                }
            }
        }


        ACC_PART_FLOAT1 :for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            ACC_PART_FLOAT2 : for(int z = 0; z < SPMM_BLOCK; z++){
            #pragma HLS UNROLL
            
                FTYPE acc_part_float = acc_part[0][j][z];
                acc2[j][z] = acc_part_float;
            }
        }
    #endif

    #ifdef EIGHTBIT

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        int BM = M[SPMM_BLOCK-1];

        int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
        M_aux[0] = 0;
        
        for(int j = 1; j < SPMM_BLOCK+1; j++){
            #pragma HLS UNROLL
            
            M_aux[j] = M[j-1];
        }

        DSP_LOOP_SPMM: for(int i = 0; i < BM; i+=1){
            #pragma HLS PIPELINE
                
            FTYPE v = A_fifo.read();
            int ci;
            ci = col_indices_fifo.read();
        
            dsp_kernel_int_fea(v, b_block, ci,   acc);




            for(int j = 0; j < B_WIDTH_BLOCK; j++){
                #pragma HLS UNROLL
                
                for(int z = 0; z < SPMM_BLOCK; z++){
                    #pragma HLS UNROLL
                    
                    if (i >= M_aux[z] && i < M_aux[z + 1])
                        acc2[j][z] += acc[j];
                
                }//z loop
            }//j loop
        } //i loop

    #endif
}

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE2_4 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute2_4(bool relu,
                int block_size,
                
                
                
                
                int first_row,
                int row_count,
                
                hls::stream<ATYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                ITYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],
                ITYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],

                ITYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],
                ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],

                hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],
                
                int B_index,
                int B_index_loop,
                int tail)
{


	    DTYPE A_accel[A_WIDTH];
		ITYPE acc[B_WIDTH_BLOCK];
		ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];

		#pragma HLS ARRAY_PARTITION variable=acc complete
		#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0



        int B_WIDTH_INT;
        ITYPE C_fifo_val;

        if (B_index < (B_index_loop-1))
            B_WIDTH_INT = B_WIDTH_BLOCK;
        else
            B_WIDTH_INT = tail;

		for(int A_index = 0; A_index < row_count; A_index += SPMM_BLOCK){
            
            // computing
			LOOP_ACC21: for(int j = 0; j < B_WIDTH_BLOCK; j++){
				#pragma HLS UNROLL
				
                LOOP_ACC22 : for(int i = 0; i < SPMM_BLOCK; i++){
					#pragma HLS UNROLL
					
                    acc2[j][i] = 0;
				}
			}

			int rnnz[SPMM_BLOCK];
			int crows = 0;
			
            LOOP_RNNZ:for(int i = 0; i < SPMM_BLOCK; i++){
	            #pragma HLS UNROLL
				
                rnnz[i] = rnnz_fifo[i].read();
				
                if ((A_index + i) < row_count)
				    crows++;

			}

			dsp_kernel_wrapper_adj_4(block_size, rnnz, A_fifo, col_indices_fifo, 
                                     B_accel1, B_accel2, B_accel3, B_accel4,
                                       acc2);


			LOOP_C_BUF1:for(int j = 0; j < B_WIDTH_BLOCK; j++){
                #pragma HLS UNROLL
                
                #if (USE_TAIL == 1)
                if (j < B_WIDTH_INT)
                #endif
                {
                    #ifdef simulation
                        if (acc2[j] < acc2_adj_min)
                            acc2_adj_min = acc2[j];
                        else if (acc2[j] > acc2_adj_max)
                            acc2_adj_max = acc2[j];
                    #endif
                    
                    
                    LOOP_C_BUF2 : for(int i = 0; i < SPMM_BLOCK; i++){
                        #pragma HLS UNROLL
                        
                        if (i < crows)
                            #if (USE_SBLOCKS == 1)
                                C_fifo[j][i].write(acc2[j][i]);
                            #endif
                            
                            #if (USE_SBLOCKS == 0)
                                    if (acc2[j][i] > 0 || relu == 0)
                                    C_fifo_val = acc2[j][i];
                                else
                                    C_fifo_val = 0.0;
                                
                                C_fifo[j][0].write(C_fifo_val);
                            #endif
                    }
                }
			}
        } // A_index loop
}



// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE2_2 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute2_2(int block_size,
                
                
                
                
                int first_row,
                int row_count,
                
                hls::stream<ATYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                ITYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],
                ITYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],

                ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],

                hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],
                
                int B_index,
                int B_index_loop,
                int tail)
{


    DTYPE A_accel[A_WIDTH];
    ITYPE acc[B_WIDTH_BLOCK];
    ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
    
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

    int B_WIDTH_INT;

    if (B_index < (B_index_loop-1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;

    for(int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK){

        //computing
        LOOP_ACC21: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            LOOP_ACC22 : for(int i = 0; i < SPMM_BLOCK; i++){
                #pragma HLS UNROLL
                
                acc2[j][i] = 0;
            }
        }

        int rnnz[SPMM_BLOCK];
        int crows = 0;
        
        LOOP_RNNZ :for(int i = 0; i < SPMM_BLOCK; i++){
            #pragma HLS UNROLL
            
            rnnz[i] = rnnz_fifo[i].read();
            
            if ((A_index+i)<row_count)
                crows++;

        }

        dsp_kernel_wrapper_adj_2(block_size, rnnz, A_fifo,col_indices_fifo, B_accel1,
                                 B_accel2,   acc2);


        LOOP_C_BUF1: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            #if (USE_TAIL == 1)
            if (j < B_WIDTH_INT)
            #endif
            {
                #ifdef simulation
                    if (acc2[j] < acc2_adj_min)
                        acc2_adj_min = acc2[j];
                    else if (acc2[j] > acc2_adj_max)
                        acc2_adj_max = acc2[j];
                #endif
                
                LOOP_C_BUF2 : for(int i = 0; i < SPMM_BLOCK; i++){
                    #pragma HLS UNROLL
                    
                    if (i < crows){
                        #if (USE_SBLOCKS == 1)
                            C_fifo[j][i].write(acc2[j][i]);
                        #endif
                        #if (USE_SBLOCKS == 0)
                            C_fifo[j][0].write(acc2[j][i]);
                        #endif
                    }
                }
            }
        }
    } // A_index loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE2_1 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute2_1(bool relu,
                int block_size,
                
                
                
                
                int first_row,
                int row_count,
                
                hls::stream<ATYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                ITYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],

                hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],
                
                int B_index,
                int B_index_loop,
                int tail)
{


	DTYPE A_accel[A_WIDTH];
	ITYPE acc[B_WIDTH_BLOCK];
	ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
	
    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

    int B_WIDTH_INT;

    if (B_index < (B_index_loop-1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;

	for(int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK){

		//computing
		LOOP_ACC21: for(int j = 0; j < B_WIDTH_BLOCK; j++){
			#pragma HLS UNROLL
			
            LOOP_ACC22 : for(int i = 0; i < SPMM_BLOCK; i++){
				#pragma HLS UNROLL
				
                acc2[j][i] = 0;
			}
		}

		int rnnz[SPMM_BLOCK];
		int crows = 0;
		
        LOOP_RNNZ :for(int i = 0; i < SPMM_BLOCK; i++){
            #pragma HLS UNROLL
			
            rnnz[i] = rnnz_fifo[i].read();
			
            if ((A_index+i)<row_count)
			    crows++;
		}

		dsp_kernel_wrapper_adj_1(block_size, rnnz, A_fifo, col_indices_fifo, 
                                 B_accel1,   acc2);


		LOOP_C_BUF1: for(int j = 0; j < B_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
            
            #if (USE_TAIL == 1)
            if (j < B_WIDTH_INT)
            #endif
            {
                #ifdef simulation
                    if (acc2[j] < acc2_adj_min)
                        acc2_adj_min = acc2[j];
                    else if (acc2[j] > acc2_adj_max)
                        acc2_adj_max = acc2[j];
                #endif


                LOOP_C_BUF2:for(int i = 0; i < SPMM_BLOCK; i++){
                    #pragma HLS UNROLL
                    
                    if (i < crows){
                        #if (USE_SBLOCKS == 1)
                            C_fifo[j][i].write(acc2[j][i]);
                        #endif
                        
                        #if (USE_SBLOCKS == 0)
                            if (acc2[j][i] > 0 || relu == 0)
                                C_fifo_val = acc2[j][i];
                            else
                                C_fifo_val = 0.0;
                            
                            C_fifo[j][0].write(C_fifo_val);
                        #endif
                    }
                }
            }
		}
    } // A_index loop
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE1_1 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute1_1(bool gemm_mode,
                
                
                
                
                int first_row,
                int row_count,
                
                hls::stream<FTYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],
                ITYPE C_buf1[B_HEIGHT][B_WIDTH_BLOCK],

                int B_index,
                int B_index_loop,
                int tail)
{

	ITYPE acc[B_WIDTH_BLOCK];
	ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];

	#pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0 //all dimensions are partitioned

    int B_WIDTH_INT;

    if (B_index < (B_index_loop - 1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;

	for(int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK){

		//computing
		LOOP_ACC21:for(int j = 0; j < C_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
			
            LOOP_ACC22:for(int i = 0; i < SPMM_BLOCK; i++){
                #pragma HLS UNROLL
			    
                acc2[j][i] = 0;
			}
		}

		int rnnz[SPMM_BLOCK];
		
        LOOP_RNNZ:for(int i = 0; i < SPMM_BLOCK; i++){
            #pragma HLS UNROLL
			
			rnnz[i] = rnnz_fifo[i].read();
		}

		dsp_kernel_wrapper_fea(gemm_mode, rnnz, A_fifo, col_indices_fifo, B_accel, acc2);

		LOOP_C_BUF1:for(int j = 0; j < C_WIDTH_BLOCK; j++){
	        #pragma HLS UNROLL
			
            #if (USE_TAIL == 1)
			if (j < B_WIDTH_INT)
			#endif
			{
				
				#ifdef simulation
                    if (acc2[j] < acc2_fea_min)
                        acc2_fea_min = acc2[j];
                    else if (acc2[j] > acc2_fea_max)
                        acc2_fea_max = acc2[j];
				#endif
				
                LOOP_C_BUF2:for(int i = 0; i < SPMM_BLOCK; i++){
					#pragma HLS UNROLL

					C_buf1[A_index + i][j] = acc2[j][i];
				}
			
			}
		}
    } // A_index loop
}



// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE1_2 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute1_2(bool gemm_mode,
                
                
                
                int first_row,
                int row_count,
                
                hls::stream<FTYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],
                ITYPE C_buf1[B_HEIGHT/4][B_WIDTH_BLOCK],
                ITYPE C_buf2[B_HEIGHT/4][B_WIDTH_BLOCK],

                ITYPE C_buf4[B_HEIGHT/4][B_WIDTH_BLOCK],

                int B_index,
                int B_index_loop,
                int tail)
{


	ITYPE acc[B_WIDTH_BLOCK];
    ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];

	#pragma HLS ARRAY_PARTITION variable=acc complete	
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0 //all dimensions are partitioned

    int B_WIDTH_INT;

    if (B_index < (B_index_loop - 1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;

	for(int A_index = 0; A_index < row_count; A_index += SPMM_BLOCK){

		//computing
		LOOP_ACC21:for(int j = 0; j < C_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
		    
            LOOP_ACC22 : for(int i = 0; i < SPMM_BLOCK; i++){
                #pragma HLS UNROLL
				acc2[j][i] = 0;
			}
		}

		int rnnz[SPMM_BLOCK];
		
        LOOP_RNNZ:for(int i = 0; i < SPMM_BLOCK; i++){
            #pragma HLS UNROLL

			rnnz[i] = rnnz_fifo[i].read();
		}

 		
		dsp_kernel_wrapper_fea(gemm_mode, rnnz, A_fifo, col_indices_fifo, B_accel,   acc2);

		LOOP_C_BUF1:for(int j = 0; j < C_WIDTH_BLOCK; j++){
	        #pragma HLS UNROLL
			
            #if (USE_TAIL == 1)
			if (j < B_WIDTH_INT)
			#endif
			{
				
				#ifdef simulation
                    if (acc2[j] < acc2_fea_min)
                        acc2_fea_min = acc2[j];
                    else if (acc2[j] > acc2_fea_max)
                        acc2_fea_max = acc2[j];
				#endif

				LOOP_C_BUF2 : for(int i = 0; i < SPMM_BLOCK; i++){
	                #pragma HLS UNROLL
					
                    C_buf1[A_index + i][j] = acc2[j][i];
					C_buf2[A_index + i][j] = acc2[j][i];
				}
	

			}
		}
    } //A_index loop
}

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						 COMPUTE1_4 								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void compute1_4(bool gemm_mode,
                
                

                int first_row,
                int row_count,
                
                hls::stream<FTYPE> &A_fifo,
                hls::stream<int> &col_indices_fifo,
                hls::stream<int> rnnz_fifo[SPMM_BLOCK],
                
                BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],
                
                ITYPE C_buf1[B_HEIGHT/4][B_WIDTH_BLOCK],
                ITYPE C_buf2[B_HEIGHT/4][B_WIDTH_BLOCK],
                ITYPE C_buf3[B_HEIGHT/4][B_WIDTH_BLOCK],
                ITYPE C_buf4[B_HEIGHT/4][B_WIDTH_BLOCK],
                
                int B_index,
                int B_index_loop,
                int tail)
    
    {



	ITYPE acc[B_WIDTH_BLOCK];
	ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
	
    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0 //all dimensions are partitioned

    int B_WIDTH_INT;

    if (B_index < (B_index_loop-1))
        B_WIDTH_INT = B_WIDTH_BLOCK;
    else
        B_WIDTH_INT = tail;


	for(int A_index = 0; A_index < row_count; A_index += SPMM_BLOCK){



		LOOP_ACC21:for(int j = 0; j < C_WIDTH_BLOCK; j++){
            #pragma HLS UNROLL
			
            LOOP_ACC22:for(int i = 0; i < SPMM_BLOCK; i++){
                #pragma HLS UNROLL
				
                acc2[j][i] = 0;
			}
		}

		int rnnz[SPMM_BLOCK];
		
        LOOP_RNNZ:for(int i = 0; i < SPMM_BLOCK; i++){
            #pragma HLS UNROLL
			
			rnnz[i] = rnnz_fifo[i].read();
		}

 		

		dsp_kernel_wrapper_fea(gemm_mode, rnnz, A_fifo, col_indices_fifo, B_accel,   acc2);

		LOOP_C_BUF1:for(int j = 0; j < C_WIDTH_BLOCK; j++){
			
	        #pragma HLS UNROLL
			
            #if (USE_TAIL == 1)
			if (j < B_WIDTH_INT)
			#endif
			{
				#ifdef simulation
                    if (acc2[j] < acc2_fea_min)
                        acc2_fea_min = acc2[j];
                    else if (acc2[j] > acc2_fea_max)
                        acc2_fea_max = acc2[j];
				#endif

                
				LOOP_C_BUF2:for(int i = 0; i < SPMM_BLOCK; i++){
	                
					C_buf1[A_index + i][j] = acc2[j][i];
					C_buf2[A_index + i][j] = acc2[j][i];
					C_buf3[A_index + i][j] = acc2[j][i];
					C_buf4[A_index + i][j] = acc2[j][i];
				}
			}
		}
    }
}
// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						FEATURES LOOP								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void loop_fea(  bool gemm_mode,
                int *rowPtr_fea1,
                int *rowPtr_fea2,
                int *rowPtr_fea3,
                int *rowPtr_fea4,

                int *columnIndex_fea1,
                int *columnIndex_fea2,
                int *columnIndex_fea3,
                int *columnIndex_fea4,

                FTYPE *values_fea1,
                FTYPE *values_fea2,
                FTYPE *values_fea3,
                FTYPE *values_fea4,

                BTYPE* B,

                int N_fea,
                int M_fea,
                
                

                hls::stream_of_blocks<buf> &C_buffer11,
                hls::stream_of_blocks<buf> &C_buffer12,

                hls::stream_of_blocks<buf> &C_buffer13,
                hls::stream_of_blocks<buf> &C_buffer14,

                hls::stream_of_blocks<buf> &C_buffer21,
                hls::stream_of_blocks<buf> &C_buffer22,

                hls::stream_of_blocks<buf> &C_buffer23,
                hls::stream_of_blocks<buf> &C_buffer24,

                hls::stream_of_blocks<buf> &C_buffer31,
                hls::stream_of_blocks<buf> &C_buffer32,

                hls::stream_of_blocks<buf> &C_buffer33,
                hls::stream_of_blocks<buf> &C_buffer34,

                hls::stream_of_blocks<buf> &C_buffer41,
                hls::stream_of_blocks<buf> &C_buffer42,

                hls::stream_of_blocks<buf> &C_buffer43,
                hls::stream_of_blocks<buf> &C_buffer44,

                int B_index_loop,
                int tail)
{


    BTYPE B_accel1[B_HEIGHT][B_WIDTH_BLOCK];
    BTYPE B_accel2[B_HEIGHT][B_WIDTH_BLOCK];
    BTYPE B_accel3[B_HEIGHT][B_WIDTH_BLOCK];
    BTYPE B_accel4[B_HEIGHT][B_WIDTH_BLOCK];
    
    hls::stream<int> rnnz_fifo_fea1[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_fea2[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_fea3[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_fea4[SPMM_BLOCK];
    
    hls::stream<FTYPE> A_fifo_fea1;
    hls::stream<FTYPE> A_fifo_fea2;
    hls::stream<FTYPE> A_fifo_fea3;
    hls::stream<FTYPE> A_fifo_fea4;
    hls::stream<FTYPE> A_fifo_fea1_out;
    
    hls::stream<bool> exit_loop;
    
    hls::stream<int>  col_indices_fifo_fea1;
    hls::stream<int>  col_indices_fifo_fea2;
    hls::stream<int>  col_indices_fifo_fea3;
    hls::stream<int>  col_indices_fifo_fea4;
    
    // ==================================================================== //
    // ==================================================================== //
    #pragma HLS array_partition variable=B_accel1 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=B_accel2 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=B_accel3 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=B_accel4 block factor= BLOCK/2 dim=2

    #pragma HLS STREAM variable=rnnz_fifo_fea1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_fea2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_fea3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_fea4 depth=FIFO_DEPTH dim=1
    
    #pragma HLS STREAM variable=A_fifo_fea1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_fea2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_fea3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_fea4 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_fea1 depth=FIFO_DEPTH dim=1
    
    #pragma HLS STREAM variable=exit_loop depth=FIFO_DEPTH dim=1
    
    #pragma HLS STREAM variable=col_indices_fifo_fea1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_fea2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_fea3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_fea4 depth=FIFO_DEPTH dim=1
    // ==================================================================== //
    // ==================================================================== //
	
    int B_WIDTH_INT;


	LOOP_FEA:for(int B_index = 0; B_index < B_index_loop; B_index++){

    	#pragma HLS DATAFLOW

        if (B_index < (B_index_loop-1))
            B_WIDTH_INT = B_WIDTH_BLOCK;
        else
            B_WIDTH_INT = tail;





		// These are the weights
		#if FEA_THREADS == 1

	  	    hls::write_lock<buf> C_fea11(C_buffer11);

	  		for(int j = 0; j < B_WIDTH_INT; j++){
	  			for(int i = 0; i < M_fea; i++){
                    
                    #pragma HLS PIPELINE
                    
                    BTYPE B_accel_temp = B[i +(j * M_fea) + (B_index * B_WIDTH_BLOCK * M_fea)];
                    B_accel1[i][j] = B_accel_temp;
                }
	  		}

            int first_row1;
            int row_count1;

            int N_fea_block = N_fea;
            int N_fea_rest = 0;
            row_count1 = N_fea_block;

            first_row1 = 0;

            reada1( gemm_mode, M_fea, first_row1, row_count1, A_fifo_fea1, 
                    col_indices_fifo_fea1, rnnz_fifo_fea1, B_index_loop, tail, 
                    rowPtr_fea1, columnIndex_fea1, values_fea1);


            compute1_1(gemm_mode,   first_row1, row_count1, 
                       A_fifo_fea1,  col_indices_fifo_fea1,  rnnz_fifo_fea1, B_accel1, C_fea11, 
                       B_index,  B_index_loop,  tail);
		#endif

        #if FEA_THREADS == 2



	        hls::write_lock<buf> C_fea11(C_buffer11);
		
            #if ADJ_THREADS == 2
                hls::write_lock<buf> C_fea12(C_buffer12);
                hls::write_lock<buf> C_fea22(C_buffer22);
            #endif
		   
		    hls::write_lock<buf> C_fea21(C_buffer21);


			for(int j = 0; j < B_WIDTH_INT; j++){
				for(int i = 0; i < M_fea; i++){
                    
                    #pragma HLS PIPELINE
                    
                    BTYPE B_accel_temp = B[i + (j * M_fea) + (B_index * B_WIDTH_BLOCK *M_fea)];
                    B_accel1[i][j] = B_accel_temp;
                    B_accel2[i][j] = B_accel_temp;
                   
                }
            }

            int first_row1,first_row2;
            int row_count1,row_count2;

            int N_fea_block = N_fea / 2;
            int N_fea_rest = N_fea % 2;
            row_count1 = N_fea_block;
            row_count2 = N_fea_block + N_fea_rest;
            first_row1 = 0;
            first_row2 = N_fea_block;

            
            reada1(gemm_mode, M_fea, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, 
                   rnnz_fifo_fea1, B_index_loop, tail, rowPtr_fea1, columnIndex_fea1, values_fea1);
            
            reada1(gemm_mode, M_fea, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, 
                   rnnz_fifo_fea2, B_index_loop, tail, rowPtr_fea2, columnIndex_fea2, values_fea2);

            #if ADJ_THREADS == 2
                compute1_2(gemm_mode,   first_row1, row_count1, 
                            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11, C_fea12,
                            B_index, B_index_loop, tail);
                
                compute1_2(gemm_mode,   first_row2, row_count2, 
                            A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21, C_fea22,
                            B_index, B_index_loop, tail);
            #endif

            #if ADJ_THREADS == 1

                compute1_1(gemm_mode,   first_row1, row_count1, 
                            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11,
                            B_index, B_index_loop, tail);

                compute1_1(gemm_mode,   first_row2, row_count2, 
                ‍‍‍‍‍           A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21,
                            B_index, B_index_loop, tail);
            #endif

        #endif


        #if FEA_THREADS == 4

            #if ADJ_THREADS == 4
                hls::write_lock<buf> C_fea11(C_buffer11);
                hls::write_lock<buf> C_fea12(C_buffer12);
                hls::write_lock<buf> C_fea13(C_buffer13);
                hls::write_lock<buf> C_fea14(C_buffer14);

                hls::write_lock<buf> C_fea21(C_buffer21);
                hls::write_lock<buf> C_fea22(C_buffer22);
                hls::write_lock<buf> C_fea23(C_buffer23);
                hls::write_lock<buf> C_fea24(C_buffer24);

                hls::write_lock<buf> C_fea31(C_buffer31);
                hls::write_lock<buf> C_fea32(C_buffer32);
                hls::write_lock<buf> C_fea33(C_buffer33);
                hls::write_lock<buf> C_fea34(C_buffer34);

                hls::write_lock<buf> C_fea41(C_buffer41);
                hls::write_lock<buf> C_fea42(C_buffer42);
                hls::write_lock<buf> C_fea43(C_buffer43);
                hls::write_lock<buf> C_fea44(C_buffer44);
            #endif

            #if ADJ_THREADS == 2

                hls::write_lock<buf> C_fea11(C_buffer11);
                hls::write_lock<buf> C_fea12(C_buffer12);
                
                hls::write_lock<buf> C_fea21(C_buffer21);
                hls::write_lock<buf> C_fea22(C_buffer22);
                
                hls::write_lock<buf> C_fea31(C_buffer31);
                hls::write_lock<buf> C_fea32(C_buffer32);
                
                hls::write_lock<buf> C_fea41(C_buffer41);
                hls::write_lock<buf> C_fea42(C_buffer42);

            #endif

            

            for(int j = 0; j < B_WIDTH_INT; j++){
                for(int i = 0; i < M_fea; i++){
                                
                    #pragma HLS PIPELINE
                    
                    BTYPE B_accel_temp = B[i + (j * M_fea) + (B_index * B_WIDTH_BLOCK * M_fea)];
                    
                    B_accel1[i][j] = B_accel_temp;
                    B_accel2[i][j] = B_accel_temp;
                    B_accel3[i][j] = B_accel_temp;
                    B_accel4[i][j] = B_accel_temp;
                }
            }



            


            int first_row1,first_row2,first_row3,first_row4;
            int row_count1,row_count2,row_count3,row_count4;


            int N_fea_block = N_fea / 4;
            int N_fea_rest = N_fea % 4;
            
            row_count1 = N_fea_block;
            row_count2 = N_fea_block;
            row_count3 = N_fea_block;
            row_count4 = N_fea_block+N_fea_rest;
            
            first_row1 = 0;
            first_row2 = N_fea_block;
            first_row3 = 2 * N_fea_block;
            first_row4 = 3 * N_fea_block;


                
            reada1(gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,B_index_loop,tail,
            rowPtr_fea1,columnIndex_fea1,values_fea1);
            
            reada1(gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,B_index_loop,tail,
            rowPtr_fea2,columnIndex_fea2,values_fea2);
            
            reada1(gemm_mode,M_fea,first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,B_index_loop,tail,
            rowPtr_fea3,columnIndex_fea3,values_fea3);
            
            reada1(gemm_mode,M_fea,first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,B_index_loop,tail,
            rowPtr_fea4,columnIndex_fea4,values_fea4);


            

            #if ADJ_THREADS == 4

                compute1_4(gemm_mode,   first_row1, row_count1,A_fifo_fea1, 
                           col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11, C_fea12, C_fea13, 
                           C_fea14, B_index, B_index_loop, tail);
                compute1_4(gemm_mode,   first_row2, row_count2,A_fifo_fea2, 
                           col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21, C_fea22, C_fea23, 
                           C_fea24, B_index, B_index_loop, tail);
                compute1_4(gemm_mode,   first_row3, row_count3,A_fifo_fea3, 
                           col_indices_fifo_fea3, rnnz_fifo_fea3, B_accel3, C_fea31, C_fea32, C_fea33, 
                           C_fea34, B_index, B_index_loop, tail);
                compute1_4(gemm_mode,   first_row4, row_count4,A_fifo_fea4, 
                           col_indices_fifo_fea4, rnnz_fifo_fea4, B_accel4, C_fea41, C_fea42, C_fea43, 
                           C_fea44, B_index, B_index_loop, tail);

            #endif

            #if ADJ_THREADS == 2

                compute1_2( gemm_mode,   first_row1,row_count1,
                            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11, 
                            C_fea12, B_index, B_index_loop, tail);
                compute1_2( gemm_mode,   first_row2,row_count2,
                            A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21, 
                            C_fea22, B_index, B_index_loop, tail);
                compute1_2( gemm_mode,   first_row3,row_count3,
                            A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3, B_accel3, C_fea31, 
                            C_fea32, B_index, B_index_loop, tail);
                compute1_2( gemm_mode,   first_row4,row_count4,
                            A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4, B_accel4, C_fea41, 
                            C_fea42, B_index, B_index_loop, tail);

            #endif
        #endif
	}


}

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 					     LOOP ADJ  								        //
// 																	    //
// ==================================================================== //
// ==================================================================== //

void loop_adj(bool relu, 
            int *rowPtr_adj1, 
            int *rowPtr_adj2, 
            int *rowPtr_adj3, 
            int *rowPtr_adj4, 

            int *columnIndex_adj1, 
            int *columnIndex_adj2, 
            int *columnIndex_adj3, 
            int *columnIndex_adj4, 

            ATYPE *values_adj1, 
            ATYPE *values_adj2, 
            ATYPE *values_adj3, 
            ATYPE *values_adj4, 

            int N_adj, 
            int M_adj, 
            int P_w, 
            
            hls::stream_of_blocks<buf> &C_buffer11, 
            hls::stream_of_blocks<buf> &C_buffer12, 

            hls::stream_of_blocks<buf> &C_buffer13, 
            hls::stream_of_blocks<buf> &C_buffer14, 

            hls::stream_of_blocks<buf> &C_buffer21, 
            hls::stream_of_blocks<buf> &C_buffer22, 

            hls::stream_of_blocks<buf> &C_buffer23, 
            hls::stream_of_blocks<buf> &C_buffer24, 

            hls::stream_of_blocks<buf> &C_buffer31, 
            hls::stream_of_blocks<buf> &C_buffer32, 

            hls::stream_of_blocks<buf> &C_buffer33, 
            hls::stream_of_blocks<buf> &C_buffer34, 

            hls::stream_of_blocks<buf> &C_buffer41, 
            hls::stream_of_blocks<buf> &C_buffer42, 

            hls::stream_of_blocks<buf> &C_buffer43, 
            hls::stream_of_blocks<buf> &C_buffer44, 

            int B_index_loop, 
            int tail, 

            DTYPE* D1, 
            DTYPE* D2, 
            DTYPE* D3, 
            DTYPE* D4)

{

    hls::stream<int> rnnz_fifo_adj1[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_adj2[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_adj3[SPMM_BLOCK];
    hls::stream<int> rnnz_fifo_adj4[SPMM_BLOCK];

    hls::stream<ATYPE> A_fifo_adj1;
    hls::stream<ATYPE> A_fifo_adj2;
    hls::stream<ATYPE> A_fifo_adj3;
    hls::stream<ATYPE> A_fifo_adj4;

    hls::stream<int>  col_indices_fifo_adj1;
    hls::stream<int>  col_indices_fifo_adj2;
    hls::stream<int>  col_indices_fifo_adj3;
    hls::stream<int>  col_indices_fifo_adj4;

    hls::stream<DTYPE> D_fifo1[C_WIDTH_BLOCK][SPMM_BLOCK];
    hls::stream<DTYPE> D_fifo2[C_WIDTH_BLOCK][SPMM_BLOCK];
    hls::stream<DTYPE> D_fifo3[C_WIDTH_BLOCK][SPMM_BLOCK];
    hls::stream<DTYPE> D_fifo4[C_WIDTH_BLOCK][SPMM_BLOCK];


    // ==================================================================== //
    // ==================================================================== //
    #pragma HLS STREAM variable=rnnz_fifo_adj1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_adj2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_adj3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=rnnz_fifo_adj4 depth=FIFO_DEPTH dim=1
    
    #pragma HLS STREAM variable=A_fifo_adj1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_adj2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_adj3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=A_fifo_adj4 depth=FIFO_DEPTH dim=1
    
    #pragma HLS STREAM variable=col_indices_fifo_adj1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_adj2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_adj3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=col_indices_fifo_adj4 depth=FIFO_DEPTH dim=1

    #pragma HLS STREAM variable=D_fifo1 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=D_fifo2 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=D_fifo3 depth=FIFO_DEPTH dim=1
    #pragma HLS STREAM variable=D_fifo4 depth=FIFO_DEPTH dim=1
    // ==================================================================== //
    // ==================================================================== //



    LOOP_ADJ:for(int B_index = 0; B_index < B_index_loop; B_index++){

        #pragma HLS DATAFLOW

        #if ADJ_THREADS == 1

            hls::read_lock<buf> C_adj11(C_buffer11);


            #if FEA_THREADS == 2
                hls::read_lock<buf> C_adj21(C_buffer21);
            #endif

            int first_row1;
            int row_count1;

            int N_adj_block = N_adj / ADJ_THREADS;
            int N_adj_block_compute = N_adj / FEA_THREADS; // In compute2 each block only contains  N_adj/FEA_THREADS elements

            row_count1 = N_adj_block;
            first_row1 = 0;

            reada2(first_row1, row_count1, B_index_loop, 
                tail,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, rowPtr_adj1, columnIndex_adj1, values_adj1);

            
            #if FEA_THREADS == 1
                compute2_1(relu, N_adj_block,   
                        first_row1, row_count1, A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
                        C_adj11, D_fifo1, B_index, B_index_loop, tail);
            #endif
            
            #if FEA_THREADS == 2
                compute2_2(N_adj_block_compute,   
                        first_row1, row_count1, A_fifo_adj1, col_indices_fifo_adj1, 
                        rnnz_fifo_adj1, C_adj11, C_adj21, D_fifo1, B_index, B_index_loop, tail);
            #endif

            writec(relu, first_row1, row_count1, P_w, D_fifo1, D1, B_index, B_index_loop, tail);
        #endif

        #if ADJ_THREADS = 2

            hls::read_lock<buf> C_adj11(C_buffer11);
            hls::read_lock<buf> C_adj12(C_buffer12);
            hls::read_lock<buf> C_adj21(C_buffer21);
            hls::read_lock<buf> C_adj22(C_buffer22);

            #if FEA_THREADS == 4
                hls::read_lock<buf> C_adj31(C_buffer31);
                hls::read_lock<buf> C_adj32(C_buffer32);
                hls::read_lock<buf> C_adj41(C_buffer41);
                hls::read_lock<buf> C_adj42(C_buffer42);
            #endif


            int first_row1,first_row2;
            int row_count1,row_count2;

            int N_adj_block = N_adj / ADJ_THREADS;
            int N_adj_rest = N_adj % ADJ_THREADS;

            int N_adj_block_compute = N_adj / FEA_THREADS; // In compute2 each block only contains  N_adj/FEA_THREADS elements
            
            row_count1 = N_adj_block;
            row_count2 = N_adj_block + N_adj_rest;;
            
            first_row1 = 0;
            first_row2 = N_adj_block;

            reada2(first_row2, row_count2, B_index_loop, tail, A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, rowPtr_adj2, columnIndex_adj2, values_adj2);

            #if FEA_THREADS == 2

            compute2_2(N_adj_block_compute,   first_row1, row_count1, 
                    A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, C_adj11, C_adj21, 
                    D_fifo1, B_index, B_index_loop, tail);

            compute2_2(N_adj_block_compute,   first_row2, row_count2, 
                    A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, C_adj12, C_adj22,
                    D_fifo2, B_index, B_index_loop, tail);

            #endif

            #if FEA_THREADS == 4


            compute2_4(relu, N_adj_block_compute,   first_row1, row_count1, 
                        A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, C_adj11, C_adj21,
                        C_adj31, C_adj41, D_fifo1, B_index, B_index_loop, tail);

            compute2_4(relu, N_adj_block_compute,   first_row2, row_count2, 
                    A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, C_adj12, C_adj22, C_adj32, C_adj42,
                    D_fifo2, B_index, B_index_loop, tail);

            #endif

            writec(relu, first_row1, row_count1, P_w, D_fifo1, D1, B_index, B_index_loop, tail);
            writec(relu, first_row2, row_count2, P_w, D_fifo2, D2, B_index, B_index_loop, tail);

        #endif


        #if ADJ_THREADS = 4

            hls::read_lock<buf> C_adj11(C_buffer11);
            hls::read_lock<buf> C_adj12(C_buffer12);
            hls::read_lock<buf> C_adj13(C_buffer13);
            hls::read_lock<buf> C_adj14(C_buffer14);
            hls::read_lock<buf> C_adj21(C_buffer21);
            hls::read_lock<buf> C_adj22(C_buffer22);
            hls::read_lock<buf> C_adj23(C_buffer23);
            hls::read_lock<buf> C_adj24(C_buffer24);
            hls::read_lock<buf> C_adj31(C_buffer31);
            hls::read_lock<buf> C_adj32(C_buffer32);
            hls::read_lock<buf> C_adj33(C_buffer33);
            hls::read_lock<buf> C_adj34(C_buffer34);
            hls::read_lock<buf> C_adj41(C_buffer41);
            hls::read_lock<buf> C_adj42(C_buffer42);
            hls::read_lock<buf> C_adj43(C_buffer43);
            hls::read_lock<buf> C_adj44(C_buffer44);

            int first_row1, first_row2, first_row3, first_row4;
            int row_count1, row_count2, row_count3, row_count4;

            int N_adj_block = N_adj / 4;
            int N_adj_rest = N_adj % 4;
            
            row_count1 = N_adj_block;
            row_count2 = N_adj_block;
            row_count3 = N_adj_block;
            row_count4 = N_adj_block+N_adj_rest;
            
            first_row1 = 0;
            first_row2 = N_adj_block;
            first_row3 = 2 * N_adj_block;
            first_row4 = 3 * N_adj_block;

            reada2(first_row1, row_count1, B_index_loop, tail, A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, rowPtr_adj1, columnIndex_adj1, values_adj1);
            reada2(first_row2, row_count2, B_index_loop, tail, A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, rowPtr_adj2, columnIndex_adj2, values_adj2);
            reada2(first_row3, row_count3, B_index_loop, tail, A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3, rowPtr_adj3, columnIndex_adj3, values_adj3);
            reada2(first_row4, row_count4, B_index_loop, tail, A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4, rowPtr_adj4, columnIndex_adj4, values_adj4);

            compute2_4(relu, N_adj_block,   first_row1, row_count1, A_fifo_adj1, col_indices_fifo_adj1, 
                        rnnz_fifo_adj1, C_adj11, C_adj21, C_adj31, C_adj41, D_fifo1, B_index, B_index_loop, tail);
            compute2_4(relu, N_adj_block,   first_row2, row_count2, A_fifo_adj2, col_indices_fifo_adj2, 
                        rnnz_fifo_adj2, C_adj12, C_adj22, C_adj32, C_adj42, D_fifo2, B_index, B_index_loop, tail);
            compute2_4(relu, N_adj_block,   first_row3, row_count3, A_fifo_adj3, col_indices_fifo_adj3, 
                        rnnz_fifo_adj3, C_adj13, C_adj23, C_adj33, C_adj43, D_fifo3, B_index, B_index_loop, tail);
            compute2_4(relu, N_adj_block,   first_row4, row_count4, A_fifo_adj4, col_indices_fifo_adj4, 
                        rnnz_fifo_adj4, C_adj14, C_adj24, C_adj34, C_adj44, D_fifo4, B_index, B_index_loop, tail);

            writec(relu, first_row1, row_count1, P_w, D_fifo1, D1, B_index, B_index_loop, tail);
            writec(relu, first_row2, row_count2, P_w, D_fifo2, D2, B_index, B_index_loop, tail);
            writec(relu, first_row3, row_count3, P_w, D_fifo3, D3, B_index, B_index_loop, tail);
            writec(relu, first_row4, row_count4, P_w, D_fifo4, D4, B_index, B_index_loop, tail);

        #endif
    }
}


// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						GFADES WRAPPER								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //
void gfades_wrapper(bool gemm_mode,
                    bool relu,

                    int N_adj,
                    int M_adj,
                    int M_fea,

                    int P_w,
                    BTYPE* B,

                    DTYPE* D1,
                    DTYPE* D2,
                    DTYPE* D3,
                    DTYPE* D4,

                    int array_c_adjust,
                    int B_index_loop,
                    int tail,

                    int *rowPtr_fea1,
                    int *rowPtr_fea2,
                    int *rowPtr_fea3,
                    int *rowPtr_fea4,

                    int *columnIndex_fea1,
                    int *columnIndex_fea2,
                    int *columnIndex_fea3,
                    int *columnIndex_fea4,

                    FTYPE *values_fea1,
                    FTYPE *values_fea2,
                    FTYPE *values_fea3,
                    FTYPE *values_fea4,

                    int *rowPtr_adj1,
                    int *rowPtr_adj2,
                    int *rowPtr_adj3,
                    int *rowPtr_adj4,

                    int *columnIndex_adj1,
                    int *columnIndex_adj2,
                    int *columnIndex_adj3,
                    int *columnIndex_adj4,

                    ATYPE *values_adj1,
                    ATYPE *values_adj2,
                    ATYPE *values_adj3,
                    ATYPE *values_adj4)
{


    hls::stream_of_blocks<buf> C_buffer11;
    hls::stream_of_blocks<buf> C_buffer12;
    hls::stream_of_blocks<buf> C_buffer13;
    hls::stream_of_blocks<buf> C_buffer14;
    hls::stream_of_blocks<buf> C_buffer21;
    hls::stream_of_blocks<buf> C_buffer22;
    hls::stream_of_blocks<buf> C_buffer23;
    hls::stream_of_blocks<buf> C_buffer24;
    hls::stream_of_blocks<buf> C_buffer31;
    hls::stream_of_blocks<buf> C_buffer32;
    hls::stream_of_blocks<buf> C_buffer33;
    hls::stream_of_blocks<buf> C_buffer34;
    hls::stream_of_blocks<buf> C_buffer41;
    hls::stream_of_blocks<buf> C_buffer42;
    hls::stream_of_blocks<buf> C_buffer43;
    hls::stream_of_blocks<buf> C_buffer44;


    // ==================================================================== //
    // ==================================================================== //
    #pragma HLS array_partition variable=C_buffer11 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer11 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer12 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer12 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer13 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer13 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer14 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer14 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer21 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer21 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer22 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer22 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer23 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer23 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer24 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer24 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer31 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer31 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer32 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer32 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer33 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer33 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer34 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer34 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer41 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer41 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer42 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer42 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer43 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer43 cyclic factor= SBLOCK dim=1

    #pragma HLS array_partition variable=C_buffer44 block factor= BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer44 cyclic factor= SBLOCK dim=1
    // ==================================================================== //
    // ==================================================================== //  

    int B_WIDTH_INT, a_values;

    #pragma HLS DATAFLOW

    loop_fea(gemm_mode, 
             rowPtr_fea1, rowPtr_fea2, rowPtr_fea3, rowPtr_fea4,
             columnIndex_fea1, columnIndex_fea2, columnIndex_fea3, columnIndex_fea4,
             values_fea1, values_fea2, values_fea3, values_fea4,
             B, M_adj, M_fea,  
             C_buffer11,C_buffer12,C_buffer13,C_buffer14,
             C_buffer21,C_buffer22,C_buffer23,C_buffer24,
             C_buffer31,C_buffer32,C_buffer33,C_buffer34,
             C_buffer41,C_buffer42,C_buffer43,C_buffer44,
             B_index_loop, tail);

    loop_adj(relu,
             rowPtr_adj1, rowPtr_adj2, rowPtr_adj3, rowPtr_adj4, 
             columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4, 
             values_adj1, values_adj2, values_adj3, values_adj4, 
             N_adj,  M_adj, P_w,   
             C_buffer11, C_buffer12, C_buffer13, C_buffer14, 
             C_buffer21, C_buffer22, C_buffer23, C_buffer24, 
             C_buffer31, C_buffer32, C_buffer33, C_buffer34, 
             C_buffer41, C_buffer42, C_buffer43, C_buffer44, 
             B_index_loop, tail, D1, D2, D3, D4);

    
}

// ==================================================================== //
// ==================================================================== //
// 																	    //
// 						  gfades TOP								    //
// 																	    //
// ==================================================================== //
// ==================================================================== //

void gfades(bool gemm_mode,

             int N_adj, 
             int M_adj, 
             int M_fea, 
             int P_w,
             
             BTYPE* B,
             
             DTYPE* D1, 
             DTYPE* D2, 
             DTYPE* D3,
             DTYPE* D4,
             
             int array_c_adjust,
             
             int *rowPtr_fea1,
             int *rowPtr_fea2,
             int *rowPtr_fea3,
             int *rowPtr_fea4,
             
             int *columnIndex_fea1, 
             int *columnIndex_fea2, 
             int *columnIndex_fea3, 
             int *columnIndex_fea4,
             
             FTYPE *values_fea1,
             FTYPE *values_fea2,
             FTYPE *values_fea3,
             FTYPE *values_fea4,
             
             int *rowPtr_adj1,
             int *rowPtr_adj2,
             int *rowPtr_adj3,
             int *rowPtr_adj4,

             int *columnIndex_adj1,
             int *columnIndex_adj2,
             int *columnIndex_adj3,
             int *columnIndex_adj4,
             
             ATYPE *values_adj1,
             ATYPE *values_adj2,
             ATYPE *values_adj3,
             ATYPE *values_adj4)
{

    // ==================================================================== //
    // ==================================================================== //
    #pragma HLS INTERFACE s_axilite port = return bundle = control
    #pragma HLS INTERFACE s_axilite port = N_adj bundle = control
    #pragma HLS INTERFACE s_axilite port = M_adj bundle = control
    #pragma HLS INTERFACE s_axilite port = M_fea bundle = control
    #pragma HLS INTERFACE s_axilite port = P_w bundle = control
    #pragma HLS INTERFACE s_axilite port = array_c_adjust bundle = control
    #pragma HLS INTERFACE s_axilite port = gemm_mode bundle = control
    #pragma HLS INTERFACE s_axilite port = relu bundle = control
    
    #pragma HLS INTERFACE m_axi port=rowPtr_fea1 depth=4096 offset=slave bundle = rowPtr_fea1
    #pragma HLS INTERFACE m_axi port=rowPtr_fea2 depth=4096 offset=slave bundle = rowPtr_fea2
    #pragma HLS INTERFACE m_axi port=rowPtr_fea3 depth=4096 offset=slave bundle = rowPtr_fea3
    #pragma HLS INTERFACE m_axi port=rowPtr_fea4 depth=4096 offset=slave bundle = rowPtr_fea4
    
    #pragma HLS INTERFACE m_axi port=columnIndex_fea1 depth=128000 offset=slave bundle = columnIndex_fea1
    #pragma HLS INTERFACE m_axi port=columnIndex_fea2 depth=128000 offset=slave bundle = columnIndex_fea2
    #pragma HLS INTERFACE m_axi port=columnIndex_fea3 depth=128000 offset=slave bundle = columnIndex_fea3
    #pragma HLS INTERFACE m_axi port=columnIndex_fea4 depth=128000 offset=slave bundle = columnIndex_fea4
    
    #pragma HLS INTERFACE m_axi port=values_fea1 depth=128000 offset=slave bundle = values_fea1
    #pragma HLS INTERFACE m_axi port=values_fea2 depth=128000 offset=slave bundle = values_fea2
    #pragma HLS INTERFACE m_axi port=values_fea3 depth=128000 offset=slave bundle = values_fea3
    #pragma HLS INTERFACE m_axi port=values_fea4 depth=128000 offset=slave bundle = values_fea4
    
    #pragma HLS INTERFACE m_axi port=rowPtr_adj1 depth=4096 offset=slave bundle = rowPtr_adj1
    #pragma HLS INTERFACE m_axi port=rowPtr_adj2 depth=4096 offset=slave bundle = rowPtr_adj2
    #pragma HLS INTERFACE m_axi port=rowPtr_adj3 depth=4096 offset=slave bundle = rowPtr_adj3
    #pragma HLS INTERFACE m_axi port=rowPtr_adj4 depth=4096 offset=slave bundle = rowPtr_adj4
    
    #pragma HLS INTERFACE m_axi port=columnIndex_adj1 depth=128000 offset=slave bundle = columnIndex_adj1
    #pragma HLS INTERFACE m_axi port=columnIndex_adj2 depth=128000 offset=slave bundle = columnIndex_adj2
    #pragma HLS INTERFACE m_axi port=columnIndex_adj3 depth=128000 offset=slave bundle = columnIndex_adj3
    #pragma HLS INTERFACE m_axi port=columnIndex_adj4 depth=128000 offset=slave bundle = columnIndex_adj4
    
    #pragma HLS INTERFACE m_axi port=values_adj1 depth=128000 offset=slave bundle = values_adj1
    #pragma HLS INTERFACE m_axi port=values_adj2 depth=128000 offset=slave bundle = values_adj2
    #pragma HLS INTERFACE m_axi port=values_adj3 depth=128000 offset=slave bundle = values_adj3
    #pragma HLS INTERFACE m_axi port=values_adj4 depth=128000 offset=slave bundle = values_adj4
    
    #pragma HLS INTERFACE m_axi port=B depth=128000 offset=slave bundle=B

    #pragma HLS INTERFACE m_axi port=D1 depth=128000 offset=slave bundle=D1
    #pragma HLS INTERFACE m_axi port=D2 depth=128000 offset=slave bundle=D2
    #pragma HLS INTERFACE m_axi port=D3 depth=128000 offset=slave bundle=D3
    #pragma HLS INTERFACE m_axi port=D4 depth=128000 offset=slave bundle=D4
    
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj1 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj2 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj3 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj4 bundle = control
    
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj1 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj2 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj3 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj4 bundle = control
    
    #pragma HLS INTERFACE s_axilite port=values_adj1  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_adj2  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_adj3  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_adj4  bundle = control

    #pragma HLS INTERFACE s_axilite port=columnIndex_fea1 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea2 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea3 bundle = control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea4 bundle = control
    
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea1 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea2 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea3 bundle = control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea4 bundle = control
    
    #pragma HLS INTERFACE s_axilite port=values_fea1  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_fea2  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_fea3  bundle = control
    #pragma HLS INTERFACE s_axilite port=values_fea4  bundle = control
    
    #pragma HLS INTERFACE s_axilite port=B  bundle = control
    
    #pragma HLS INTERFACE s_axilite port=D1  bundle = control
    #pragma HLS INTERFACE s_axilite port=D2  bundle = control
    #pragma HLS INTERFACE s_axilite port=D3  bundle = control
    #pragma HLS INTERFACE s_axilite port=D4  bundle = control
    // ==================================================================== //
    // ==================================================================== //

    bool relu = 0;
 

    INT32 tail = P_w % B_WIDTH_BLOCK;
    INT32 B_index_loop = (P_w / B_WIDTH_BLOCK) + 1;

    if (tail == 0){
        B_index_loop = P_w/B_WIDTH_BLOCK;
        tail = B_WIDTH_BLOCK;
    }

    int N_adj_block = N_adj / ADJ_THREADS;
    
    D2 += N_adj_block * P_w;
    D3 += 2 * N_adj_block * P_w;
    D4 += 3 * N_adj_block * P_w;

    gfades_wrapper( gemm_mode,
                    relu,

                    N_adj,
                    M_adj,
                    M_fea,
                    P_w,

                    B,
                    D1,
                    D2,
                    D3,
                    D4,

                    array_c_adjust,
                    B_index_loop,
                    tail,

                    rowPtr_fea1,
                    rowPtr_fea2,
                    rowPtr_fea3,
                    rowPtr_fea4,

                    columnIndex_fea1,
                    columnIndex_fea2,
                    columnIndex_fea3,
                    columnIndex_fea4,

                    values_fea1,
                    values_fea2,
                    values_fea3,
                    values_fea4,

                    rowPtr_adj1,
                    rowPtr_adj2,
                    rowPtr_adj3,
                    rowPtr_adj4,

                    columnIndex_adj1,
                    columnIndex_adj2,
                    columnIndex_adj3,
                    columnIndex_adj4,

                    values_adj1,
                    values_adj2,
                    values_adj3,
                    values_adj4);    
}