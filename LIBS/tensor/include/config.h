//
// Created by hslhe on 13/11/2021.
//

#ifndef TENSOR_REA_H
#define TENSOR_REA_H
#define REAL float
#if (REAL == float)
#define KREAL  "#define REAL float\n"
#else
#define KREAL  "#define REAL double\n"
#endif
#endif //TENSOR_REA_H
