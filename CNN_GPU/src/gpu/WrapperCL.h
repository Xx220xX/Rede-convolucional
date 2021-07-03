//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_WRAPPERCL_H
#define GAB_WRAPPERCL_H

#ifdef OPENCL_INTEL
#error "ESSA MACRO É OBSOLETA"
#include <CLIntel/cl.h>
#endif
#include<CL/cl.h>
#include"Kernel.h"
#include <stdio.h>


typedef struct {
    cl_platform_id platformId;
    cl_device_id device;
    cl_uint compute_units;
    cl_program program;
    cl_context context;
    size_t maxworks ;
    cl_device_type type_device;
} WrapperCL;

typedef struct{
    cl_int error;
    char msg[255];
}GPU_ERROR;


int WrapperCL_init(WrapperCL *self, const char *src);
int WrapperCL_initbyFile(WrapperCL *self,const char * filename);
void WrapperCL_release(WrapperCL *self);
void showError(int error);
cl_program  compileProgram(cl_context ct,cl_device_id dv,const char *source);


#define PERRW(e,x,contextName)if(e){fprintf(stderr,"%s: %s  error code : %d\n\t",contextName,x,e);showError(e);}
#define PERR(e,x,contextName)if(e){fprintf(stderr,"%s: %s error code : %d\n\t",contextName,x,e);showError(e);return e;}
#define PER(e,x,contextName)if(e){fprintf(stderr,"%s: %s error code: %d\n\t",contextName,x,e);showError(e);exit(e);}


#endif //GAB_WRAPPERCL_H
