//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_WRAPPERCL_H
#define GAB_WRAPPERCL_H

#include<CL/cl.h>
#include"Kernel.h"
#include <stdio.h>

#define REAL long double

typedef struct {
    cl_platform_id platformId;
    cl_device_id device;
    cl_uint compute_units;
    cl_program program;
    cl_context context;
} WrapperCL;
typedef struct{
    cl_int error;
    char msg[255];
}GPU_ERROR;
int WrapperCL_init(WrapperCL *self, const char *src);

void WrapperCL_release(WrapperCL *self);
void showError(int error);
#define PERR(e,x)if(e){fprintf(stderr,"%s error code: %d\n\t",x,e);showError(e);return e;}
#define PER(e,x)if(e){fprintf(stderr,"%s error code: %d\n\t",x,e);showError(e);exit(e);}


#endif //GAB_WRAPPERCL_H
