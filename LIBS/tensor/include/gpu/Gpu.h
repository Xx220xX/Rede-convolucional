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

typedef cl_command_queue Queue;

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
	/// Libera os recursos
	void (*release)(struct Gpu_t**self_p);
	/// retorna uma cadeia de caracteres contendo a mensagem referente ao codigo(a mensage deve ser liberada com free_mem)
	char *(*errorMsg)(int error_code);
	/// compila os kernel e salva em self.program, se compilado novamente, o anterior será apagado
	int (*compileProgram)(struct Gpu_t*self, char *program_source);
	/// compila os kernel de um arquivo e salva em self.program, se compilado novamente, o anterior será apagado
	int (*compileProgramFile)(struct Gpu_t*self, char *program_file);
	/// Obtem informações da gpu
	CLInfo (*getClInfo)(struct Gpu_t*self);
	/// Cria uma nova cl_command Queue, deve ser liberada com clCommandQueueRelease
	Queue (*Queue_new)(struct Gpu_t *self,cl_int *error);
} *Gpu, Gpu_t;

extern char *Gpu_errormsg(int error);

extern Gpu Gpu_new();

#endif