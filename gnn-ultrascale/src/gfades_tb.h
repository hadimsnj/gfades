// ==================================================================== //
// ==================================================================== //
// 	This file is part of the gFADES GNN accelerator has been written    //
//  at Linkoping University for the WASP project						//
// 						        								        //
// 	Author : Jose Nunez-Yanez											//
// ==================================================================== //
// ==================================================================== //

#ifndef KERNELMATRIXMULT_H_
#define KERNELMATRIXMULT_H_

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
             ATYPE *values_adj4);

#endif 
