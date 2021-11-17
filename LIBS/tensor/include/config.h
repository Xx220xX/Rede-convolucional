//
// Created by hslhe on 13/11/2021.
//

#ifndef TENSOR_REA_H
#define TENSOR_REA_H
#define REAL float

#if (REAL == float)
#define KREAL  "#define REAL float\n"
#define CL_REAL cl_float

#else
#define KREAL  "#define REAL double\n"
#define CL_REAL cl_double

#endif



#define  DEBUG_STACK 1
#endif //TENSOR_REA_H
