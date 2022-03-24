//
// Created by Henrique on 29/11/2021.
//
/***
 * Neste arquivo será listado todos os possiveis erros do projeto com excessão dos erros openCL
 * portanto os erros são valores positivos
 */

#ifndef GAB_ERROR_LIST_H
#define GAB_ERROR_LIST_H

#define  GAB_FAILED_OPEN_FILE 1
#define  GAB_FAILED_CREATE_FILE 2
#define  GAB_FILE_NOT_FOUD 8
#define  GAB_EXPECTED_PARAM 3
#define  GAB_NULL_POINTER_ERROR 4
#define  GAB_INDEX_OUT_OF_BOUNDS 5
#define  GAB_FAILED_ALLOC_MEM 6
#define  GAB_INVALID_PARAM 7
#define  GAB_INVALID_LAYER 9
#define  GAB_INVALID_DIVISION 10
#define  GAB_INVALID_MEMORY 11
#define  GAB_CNN_NOT_INITIALIZED 34
#define  GAB_ERRO_LUA  40


#endif //GAB_ERROR_LIST_H
