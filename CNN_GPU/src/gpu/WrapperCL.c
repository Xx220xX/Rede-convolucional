//
// Created by Xx220xX on 12/05/2020.
//

#include "gpu/WrapperCL.h"
#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>

int WrapperCL_init_file(WrapperCL *self, const char *filename) {
	FILE *f;
	f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "arquivo nao encontrado no caminho %s\n", filename);
		return -1;
	}
	char *src = 0;
	long int size = 0;
	fseek(f, 0L, SEEK_END);
	size = ftell(f);
	src = alloc_mem(size, sizeof(char));
	fseek(f, 0L, SEEK_SET);
	fread(src, sizeof(char), size, f);
	src[size - 1] = 0;
	WrapperCl_init(self, src);
	free_mem(src);
	fclose(f);
	return 0;
}

int WrapperCl_init(WrapperCL *self, const char *src) {
	cl_int error = CL_SUCCESS;
	if (self->type_device == -1) {
		self->type_device = CL_DEVICE_TYPE_GPU;
	}
	// getPlatform
	error = clGetPlatformIDs(1, &self->platformId, NULL);
	PERR(error, "falha ao pegar plataformas", "wrapper init");

	// get device
	error = clGetDeviceIDs(self->platformId, self->type_device, 1, &self->device, NULL);
	PERR(error, "falha ao pegar dispositivos", "wrapper init");

	// get size of compute
	error = clGetDeviceInfo(self->device, CL_DEVICE_MAX_COMPUTE_UNITS,
							sizeof(cl_uint), &self->compute_units, NULL);
	PERR(error, "falha ao pegar informacao do dispositivo", "wrapper init");

	// create context
	self->context = clCreateContext(NULL, 1, &self->device, 0, NULL, &error);
	PERR(error, "failed when try create context", "wrapper init");


	size_t maxLW = 1;
	int errorr = clGetDeviceInfo(self->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
	if (errorr)fprintf(stderr, "falha ao checar valor error id: %d\n", error);
	self->maxworks = maxLW;

	// compile program kernel
	self->program = compileProgram(self->context, self->device, src, &error);
	PERR(error, "failed to create program", "wrapper init");


	// build
	cl_int stt = clBuildProgram(self->program, 1, &self->device, NULL, NULL, NULL);
	if (stt != CL_SUCCESS) {
		return stt;
	}
	return 0;

}

void WrapperCL_release(WrapperCL *self) {
	clReleaseProgram(self->program);
	clReleaseDevice(self->device);
	clReleaseContext(self->context);
}

char *getClErrorWithContext(int error, char *msg, int len_msg, char *context, ...) {
	va_list args;
	va_start(args, context);
	int len = vsnprintf(msg, len_msg, context, args);
	va_end(args);
	getClError(error, msg + len, len_msg - len);
	return msg;
}

char *getClError(int error, char *msg, int len_msg) {
	switch (error) {
		case 0:
			snprintf(msg, len_msg, "%s", "SUCCESS");
			break;
		case -1:
			snprintf(msg, len_msg, "%s", "DEVICE_NOT_FOUND");
			break;
		case -2:
			snprintf(msg, len_msg, "%s", "DEVICE_NOT_AVAILABLE");
			break;
		case -3:
			snprintf(msg, len_msg, "%s", "COMPILER_NOT_AVAILABLE");
			break;
		case -4:
			snprintf(msg, len_msg, "%s", "MEM_OBJECT_ALLOCATION_FAILURE");
			break;
		case -5:
			snprintf(msg, len_msg, "%s", "OUT_OF_RESOURCES");
			break;
		case -6:
			snprintf(msg, len_msg, "%s", "OUT_OF_HOST_MEMORY");
			break;
		case -7:
			snprintf(msg, len_msg, "%s", "PROFILING_INFO_NOT_AVAILABLE");
			break;
		case -8:
			snprintf(msg, len_msg, "%s", "MEM_COPY_OVERLAP");
			break;
		case -9:
			snprintf(msg, len_msg, "%s", "IMAGE_FORMAT_MISMATCH");
			break;
		case -10:
			snprintf(msg, len_msg, "%s", "IMAGE_FORMAT_NOT_SUPPORTED");
			break;
		case -11:
			snprintf(msg, len_msg, "%s", "BUILD_PROGRAM_FAILURE");
			break;
		case -12:
			snprintf(msg, len_msg, "%s", "MAP_FAILURE");
			break;
		case -13:
			snprintf(msg, len_msg, "%s", "MISALIGNED_SUB_BUFFER_OFFSET");
			break;
		case -14:
			snprintf(msg, len_msg, "%s", "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
			break;
		case -15:
			snprintf(msg, len_msg, "%s", "COMPILE_PROGRAM_FAILURE");
			break;
		case -16:
			snprintf(msg, len_msg, "%s", "LINKER_NOT_AVAILABLE");
			break;
		case -17:
			snprintf(msg, len_msg, "%s", "LINK_PROGRAM_FAILURE");
			break;
		case -18:
			snprintf(msg, len_msg, "%s", "DEVICE_PARTITION_FAILED");
			break;
		case -19:
			snprintf(msg, len_msg, "%s", "KERNEL_ARG_INFO_NOT_AVAILABLE");
			break;
		case -30:
			snprintf(msg, len_msg, "%s", "INVALID_VALUE");
			break;
		case -31:
			snprintf(msg, len_msg, "%s", "INVALID_DEVICE_TYPE");
			break;
		case -32:
			snprintf(msg, len_msg, "%s", "INVALID_PLATFORM");
			break;
		case -33:
			snprintf(msg, len_msg, "%s", "INVALID_DEVICE");
			break;
		case -34:
			snprintf(msg, len_msg, "%s", "INVALID_CONTEXT");
			break;
		case -35:
			snprintf(msg, len_msg, "%s", "INVALID_QUEUE_PROPERTIES");
			break;
		case -36:
			snprintf(msg, len_msg, "%s", "INVALID_COMMAND_QUEUE");
			break;
		case -37:
			snprintf(msg, len_msg, "%s", "INVALID_HOST_PTR");
			break;
		case -38:
			snprintf(msg, len_msg, "%s", "INVALID_MEM_OBJECT");
			break;
		case -39:
			snprintf(msg, len_msg, "%s", "INVALID_IMAGE_FORMAT_DESCRIPTOR");
			break;
		case -40:
			snprintf(msg, len_msg, "%s", "INVALID_IMAGE_SIZE");
			break;
		case -41:
			snprintf(msg, len_msg, "%s", "INVALID_SAMPLER");
			break;
		case -42:
			snprintf(msg, len_msg, "%s", "INVALID_BINARY");
			break;
		case -43:
			snprintf(msg, len_msg, "%s", "INVALID_BUILD_OPTIONS");
			break;
		case -44:
			snprintf(msg, len_msg, "%s", "INVALID_PROGRAM");
			break;
		case -45:
			snprintf(msg, len_msg, "%s", "INVALID_PROGRAM_EXECUTABLE");
			break;
		case -46:
			snprintf(msg, len_msg, "%s", "INVALID_KERNEL_NAME");
			break;
		case -47:
			snprintf(msg, len_msg, "%s", "INVALID_KERNEL_DEFINITION");
			break;
		case -48:
			snprintf(msg, len_msg, "%s", "INVALID_KERNEL");
			break;
		case -49:
			snprintf(msg, len_msg, "%s", "INVALID_ARG_INDEX");
			break;
		case -50:
			snprintf(msg, len_msg, "%s", "INVALID_ARG_VALUE");
			break;
		case -51:
			snprintf(msg, len_msg, "%s", "INVALID_ARG_SIZE");
			break;
		case -52:
			snprintf(msg, len_msg, "%s", "INVALID_KERNEL_ARGS");
			break;
		case -53:
			snprintf(msg, len_msg, "%s", "INVALID_WORK_DIMENSION");
			break;
		case -54:
			snprintf(msg, len_msg, "%s", "INVALID_WORK_GROUP_SIZE");
			break;
		case -55:
			snprintf(msg, len_msg, "%s", "INVALID_WORK_ITEM_SIZE");
			break;
		case -56:
			snprintf(msg, len_msg, "%s", "INVALID_GLOBAL_OFFSET");
			break;
		case -57:
			snprintf(msg, len_msg, "%s", "INVALID_EVENT_WAIT_LIST");
			break;
		case -58:
			snprintf(msg, len_msg, "%s", "INVALID_EVENT");
			break;
		case -59:
			snprintf(msg, len_msg, "%s", "INVALID_OPERATION");
			break;
		case -60:
			snprintf(msg, len_msg, "%s", "INVALID_GL_OBJECT");
			break;
		case -61:
			snprintf(msg, len_msg, "%s", "INVALID_BUFFER_SIZE");
			break;
		case -62:
			snprintf(msg, len_msg, "%s", "INVALID_MIP_LEVEL");
			break;
		case -63:
			snprintf(msg, len_msg, "%s", "INVALID_GLOBAL_WORK_SIZE");
			break;
		case -64:
			snprintf(msg, len_msg, "%s", "INVALID_PROPERTY");
			break;
		case -65:
			snprintf(msg, len_msg, "%s", "INVALID_IMAGE_DESCRIPTOR");
			break;
		case -66:
			snprintf(msg, len_msg, "%s", "INVALID_COMPILER_OPTIONS");
			break;
		case -67:
			snprintf(msg, len_msg, "%s", "INVALID_LINKER_OPTIONS");
			break;
		case -68:
			snprintf(msg, len_msg, "%s", "INVALID_DEVICE_PARTITION_COUNT");
			break;
		default:
			break;
	}
	return msg;
}

void showError(int error) {
	char errormsg[EXCEPTION_MAX_MSG_SIZE];
	getClError(error, errormsg, EXCEPTION_MAX_MSG_SIZE);
	fprintf(stderr, "%s\n", errormsg);
}

cl_program compileProgram(cl_context ct, cl_device_id dv, const char *source, int *error) {
	if (!source)return NULL;
	size_t length = strlen(source);
	int __error__ = 0;
	if (!error)error = &__error__;
	cl_program program = clCreateProgramWithSource(ct, // contexto
												   1,       // numero de strings
												   &source,    // strings
												   &length, // tamanho de cada string
												   error   // error check
	);
	PERRW(*error, "failed to create program", "compile program");
	cl_int stt = clBuildProgram(program, 1, &dv, NULL, NULL, NULL);
	if (stt != CL_SUCCESS) {
		char buff[0x10000];
		clGetProgramBuildInfo(program, dv, CL_PROGRAM_BUILD_LOG, 0x10000, buff, NULL);
		fprintf(stderr, "CNN_ERROR: %s\n", buff);
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}


void printCLInfo(CLInfo cif) {
	char buff[250];
	printf("global_mem_cache_size %s\n"
		   "global_mem_cache_line_size %s\n"
		   "global_mem_size %s\n"
		   "local_mem_size %s\n"
		   "max_mem_alloc_size %s\n"
		   "max_freq_mhz %u\n"
		   "max_work_group_size %llu\n"
		   "max_work_item_sizes (%llu,%llu,%llu)\n\n",
		   printBytes(cif.global_mem_cache_size, buff),
		   printBytes(cif.global_mem_cache_line_size, buff),
		   printBytes(cif.global_mem_size, buff),
		   printBytes(cif.local_mem_size, buff),
		   printBytes(cif.max_mem_alloc_size, buff),
		   cif.max_freq_mhz,
		   cif.max_work_group_size,
		   cif.max_work_item_sizes[0],
		   cif.max_work_item_sizes[1],
		   cif.max_work_item_sizes[2]
	);
}

CLInfo getClinfo(WrapperCL *cl) {
	CLInfo cif;
	clGetDeviceInfo(cl->device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cif.global_mem_cache_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_ulong), &cif.global_mem_cache_line_size,
					NULL);
	clGetDeviceInfo(cl->device, CL_MEM_SIZE, sizeof(cl_ulong), &cif.global_mem_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &cif.local_mem_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &cif.max_freq_mhz, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &cif.max_mem_alloc_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cif.max_work_group_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &cif.max_work_item_sizes, NULL);
	return cif;
}
