#ifndef __GCN_H__
#define __GCN_H__


/*
Adj Matrix - NxN  
Fea Matrix - NxM
Wei Matrix - MxP
*/


//#define simulation

#define MAX_N    4096 
#define MAX_M    4096 
#define MAX_P    32

#define MAX_FIFO 16


#define A_HEIGHT   MAX_N
#define A_WIDTH    MAX_N

#define B_HEIGHT   MAX_N
#define B_WIDTH    MAX_M

#define C_HEIGHT   MAX_M
#define C_WIDTH    MAX_P


#define HALF
//#define FLOAT
// #define EIGHTBIT


#ifdef EIGHTBIT

	typedef ap_fixed<8, 2> ATYPE;
	typedef ap_fixed<8, 2> BTYPE;
	typedef ap_fixed<8, 2> DTYPE;
	typedef ap_fixed<8, 2> FTYPE;
	typedef ap_fixed<8, 2> ITYPE;

    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1

#endif

#ifdef HALF

	typedef half ATYPE;
	typedef half BTYPE;
	typedef half DTYPE;
	typedef half FTYPE;
	typedef half ITYPE;

	#define FTYPE_LATENCY_ADJ 4
	#define FTYPE_LATENCY_FEA 4
#endif


#ifdef FLOAT

	typedef float ATYPE;
	typedef float BTYPE;
	typedef float DTYPE;
	typedef float FTYPE;
	typedef float ITYPE;
	
	#define FTYPE_LATENCY_ADJ 6
	#define FTYPE_LATENCY_FEA 6 
#endif

#define FEA_THREADS 2
#define ADJ_THREADS 2

#define SPMM_BLOCK 1
#define USE_SBLOCKS 0


#define A_HEIGHT_BLOCK  1// 4096 //(512/4)
#define B_WIDTH_BLOCK 8 //the width of compute1 BLOCK BUFFER A*B = C 16 //32 //64 //64 //128 // 64 //64 //64 //8//8// //16//32//1//32//1//32//1// 1//32//(128/4)
#define C_WIDTH_BLOCK 8 //the width of compute2 BLOCK BUFFER C*D = F
#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 
#define B_BLOCK_PARALLEL 1

#define ENABLE_GEMM
#define ENABLE_SPMM


typedef ap_int<8>  INT8;
typedef ap_int<16> INT16;
typedef ap_int<32> INT32;
typedef ap_int<64> INT64;


#endif
