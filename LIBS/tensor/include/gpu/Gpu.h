//
// Created by Xx220xX on 12/05/2020.
//

#ifndef GAB_WRAPPERCL_H
#define GAB_WRAPPERCL_H

#ifndef free_mem
#define free_mem free
#endif
#ifndef alloc_mem
#define alloc_mem calloc
#endif


#include <stdio.h>
#include<CL/opencl.h>

typedef struct CLInfo {
	cl_ulong global_mem_cache_size;
	cl_ulong global_mem_cache_line_size;
	cl_ulong global_mem_size;
	cl_ulong local_mem_size;

	cl_uint max_freq_mhz;
	cl_ulong max_mem_alloc_size;
	size_t max_work_group_size;
	size_t max_work_item_sizes[3];
	char device_name[64];
	char hardware_version[64];
	char software_version[64];
	char openCL_version[64];

	// metodo
	char *(*json)(void *self);

} CLInfo;
/***
 * Interface para simplificar uso da API openCL
 */
typedef struct Gpu_t {
	cl_platform_id platformId;
	cl_device_id device;
	cl_uint compute_units;
	cl_program program;
	cl_context context;
	size_t maxworks;
	cl_device_type type_device;

	int error;

	void (*release)(void *self_p);

	char *(*json)(void *self);

	char *(*errorMsg)(int error_code);

	int (*compileProgram)(void *self, char *program_source);

	int (*compileProgramFile)(void *self, char *program_file);

	CLInfo (*getClInfo)(void *self);
} *Gpu, Gpu_t;

extern char *Gpu_errormsg(int error);

extern Gpu Gpu_new();

#endif