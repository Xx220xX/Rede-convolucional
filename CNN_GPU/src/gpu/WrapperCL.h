//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_WRAPPERCL_H
#define GAB_WRAPPERCL_H



#include<CL/cl.h>
#include"Kernel.h"
#include <stdio.h>


typedef struct {
	cl_platform_id platformId;
	cl_device_id device;
	cl_uint compute_units;
	cl_program program;
	cl_context context;
	size_t maxworks;
	cl_device_type type_device;
} WrapperCL;

typedef struct{
	cl_ulong global_mem_cache_size;
	cl_ulong global_mem_cache_line_size;
	cl_ulong global_mem_size;
	cl_ulong local_mem_size;

	cl_uint max_freq_mhz;
	cl_ulong max_mem_alloc_size;
	size_t max_work_group_size;
	size_t max_work_item_sizes[3];
}CLInfo;
void printCLInfo(CLInfo cif);

int WrapperCL_init(WrapperCL *self, const char *src);

int WrapperCL_initbyFile(WrapperCL *self, const char *filename);

void WrapperCL_release(WrapperCL *self);

void showError(int error);

cl_program compileProgram(cl_context ct, cl_device_id dv, const char *source);

void getClError(int error, char *msg);
CLInfo getClinfo(WrapperCL *cl);
char *printBytes(cl_ulong bytes, char buff[250]);
#define PERRW(e, x, contextName)if(e){fprintf(stderr,"%s: %s  error code : %d\n\t",contextName,x,e);showError(e);}
#define PERR(e, x, contextName)if(e){fprintf(stderr,"%s: %s error code : %d\n\t",contextName,x,e);showError(e);return e;}
#define PER(e, x, contextName)if(e){fprintf(stderr,"%s: %s error code: %d\n\t",contextName,x,e);showError(e);exit(e);}
#endif //GAB_WRAPPERCL_H
