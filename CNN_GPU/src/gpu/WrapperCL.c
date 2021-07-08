//
// Created by Xx220xX on 12/05/2020.
//

#include "WrapperCL.h"
#include<stdio.h>
#include<stdlib.h>

int WrapperCL_initbyFile(WrapperCL *self, const char *filename) {
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
	src = calloc(size, sizeof(char));
	fseek(f, 0L, SEEK_SET);
	fread(src, sizeof(char), size, f);
	src[size - 1] = 0;
	WrapperCL_init(self, src);
	free(src);
	fclose(f);
	return 0;
}

int WrapperCL_init(WrapperCL *self, const char *src) {
//    printf("%s\n",src);
	const size_t length = strlen(src);
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

	// compile program kernel
	self->program = clCreateProgramWithSource(self->context, // contexto
	                                          1,       // numero de strings
	                                          &src,    // strings
	                                          &length, // tamanho de cada string
	                                          &error   // error check
	);
	PERR(error, "failed to create program", "wrapper init");

	// build
	cl_int stt = clBuildProgram(self->program, 1, &self->device, NULL, NULL, NULL);
	if (stt != CL_SUCCESS) {
		char buff[0x10000];
		clGetProgramBuildInfo(self->program, self->device, CL_PROGRAM_BUILD_LOG, 0x10000, buff, NULL);
		fprintf(stderr, "ERROR: %s\n", buff);

		return stt;
	}
	size_t maxLW = 1;
	int errorr = clGetDeviceInfo(self->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
	if (errorr)fprintf(stderr, "falha ao checar valor error id: %d\n", error);

	self->maxworks = maxLW;
	return 0;

}

void WrapperCL_release(WrapperCL *self) {
	clReleaseProgram(self->program);
	clReleaseDevice(self->device);
	clReleaseContext(self->context);
}

void getClError(int error, char *msg) {
	switch (error) {
		case 0:
			sprintf(msg, "%s", "SUCCESS");
			break;
		case -1:
			sprintf(msg, "%s", "DEVICE_NOT_FOUND");
			break;
		case -2:
			sprintf(msg, "%s", "DEVICE_NOT_AVAILABLE");
			break;
		case -3:
			sprintf(msg, "%s", "COMPILER_NOT_AVAILABLE");
			break;
		case -4:
			sprintf(msg, "%s", "MEM_OBJECT_ALLOCATION_FAILURE");
			break;
		case -5:
			sprintf(msg, "%s", "OUT_OF_RESOURCES");
			break;
		case -6:
			sprintf(msg, "%s", "OUT_OF_HOST_MEMORY");
			break;
		case -7:
			sprintf(msg, "%s", "PROFILING_INFO_NOT_AVAILABLE");
			break;
		case -8:
			sprintf(msg, "%s", "MEM_COPY_OVERLAP");
			break;
		case -9:
			sprintf(msg, "%s", "IMAGE_FORMAT_MISMATCH");
			break;
		case -10:
			sprintf(msg, "%s", "IMAGE_FORMAT_NOT_SUPPORTED");
			break;
		case -11:
			sprintf(msg, "%s", "BUILD_PROGRAM_FAILURE");
			break;
		case -12:
			sprintf(msg, "%s", "MAP_FAILURE");
			break;
		case -13:
			sprintf(msg, "%s", "MISALIGNED_SUB_BUFFER_OFFSET");
			break;
		case -14:
			sprintf(msg, "%s", "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
			break;
		case -15:
			sprintf(msg, "%s", "COMPILE_PROGRAM_FAILURE");
			break;
		case -16:
			sprintf(msg, "%s", "LINKER_NOT_AVAILABLE");
			break;
		case -17:
			sprintf(msg, "%s", "LINK_PROGRAM_FAILURE");
			break;
		case -18:
			sprintf(msg, "%s", "DEVICE_PARTITION_FAILED");
			break;
		case -19:
			sprintf(msg, "%s", "KERNEL_ARG_INFO_NOT_AVAILABLE");
			break;
		case -30:
			sprintf(msg, "%s", "INVALID_VALUE");
			break;
		case -31:
			sprintf(msg, "%s", "INVALID_DEVICE_TYPE");
			break;
		case -32:
			sprintf(msg, "%s", "INVALID_PLATFORM");
			break;
		case -33:
			sprintf(msg, "%s", "INVALID_DEVICE");
			break;
		case -34:
			sprintf(msg, "%s", "INVALID_CONTEXT");
			break;
		case -35:
			sprintf(msg, "%s", "INVALID_QUEUE_PROPERTIES");
			break;
		case -36:
			sprintf(msg, "%s", "INVALID_COMMAND_QUEUE");
			break;
		case -37:
			sprintf(msg, "%s", "INVALID_HOST_PTR");
			break;
		case -38:
			sprintf(msg, "%s", "INVALID_MEM_OBJECT");
			break;
		case -39:
			sprintf(msg, "%s", "INVALID_IMAGE_FORMAT_DESCRIPTOR");
			break;
		case -40:
			sprintf(msg, "%s", "INVALID_IMAGE_SIZE");
			break;
		case -41:
			sprintf(msg, "%s", "INVALID_SAMPLER");
			break;
		case -42:
			sprintf(msg, "%s", "INVALID_BINARY");
			break;
		case -43:
			sprintf(msg, "%s", "INVALID_BUILD_OPTIONS");
			break;
		case -44:
			sprintf(msg, "%s", "INVALID_PROGRAM");
			break;
		case -45:
			sprintf(msg, "%s", "INVALID_PROGRAM_EXECUTABLE");
			break;
		case -46:
			sprintf(msg, "%s", "INVALID_KERNEL_NAME");
			break;
		case -47:
			sprintf(msg, "%s", "INVALID_KERNEL_DEFINITION");
			break;
		case -48:
			sprintf(msg, "%s", "INVALID_KERNEL");
			break;
		case -49:
			sprintf(msg, "%s", "INVALID_ARG_INDEX");
			break;
		case -50:
			sprintf(msg, "%s", "INVALID_ARG_VALUE");
			break;
		case -51:
			sprintf(msg, "%s", "INVALID_ARG_SIZE");
			break;
		case -52:
			sprintf(msg, "%s", "INVALID_KERNEL_ARGS");
			break;
		case -53:
			sprintf(msg, "%s", "INVALID_WORK_DIMENSION");
			break;
		case -54:
			sprintf(msg, "%s", "INVALID_WORK_GROUP_SIZE");
			break;
		case -55:
			sprintf(msg, "%s", "INVALID_WORK_ITEM_SIZE");
			break;
		case -56:
			sprintf(msg, "%s", "INVALID_GLOBAL_OFFSET");
			break;
		case -57:
			sprintf(msg, "%s", "INVALID_EVENT_WAIT_LIST");
			break;
		case -58:
			sprintf(msg, "%s", "INVALID_EVENT");
			break;
		case -59:
			sprintf(msg, "%s", "INVALID_OPERATION");
			break;
		case -60:
			sprintf(msg, "%s", "INVALID_GL_OBJECT");
			break;
		case -61:
			sprintf(msg, "%s", "INVALID_BUFFER_SIZE");
			break;
		case -62:
			sprintf(msg, "%s", "INVALID_MIP_LEVEL");
			break;
		case -63:
			sprintf(msg, "%s", "INVALID_GLOBAL_WORK_SIZE");
			break;
		case -64:
			sprintf(msg, "%s", "INVALID_PROPERTY");
			break;
		case -65:
			sprintf(msg, "%s", "INVALID_IMAGE_DESCRIPTOR");
			break;
		case -66:
			sprintf(msg, "%s", "INVALID_COMPILER_OPTIONS");
			break;
		case -67:
			sprintf(msg, "%s", "INVALID_LINKER_OPTIONS");
			break;
		case -68:
			sprintf(msg, "%s", "INVALID_DEVICE_PARTITION_COUNT");
			break;
		default:
			break;
	}
}

void showError(int error) {
	char errormsg[50];
	getClError(error, errormsg);
	fprintf(stderr, "%s\n", errormsg);
}

cl_program compileProgram(cl_context ct, cl_device_id dv, const char *source) {
	size_t length = strlen(source);
	int error = 0;
	cl_program program = clCreateProgramWithSource(ct, // contexto
	                                               1,       // numero de strings
	                                               &source,    // strings
	                                               &length, // tamanho de cada string
	                                               &error   // error check
	);
	PERRW(error, "failed to create program", "compile program");
	cl_int stt = clBuildProgram(program, 1, &dv, NULL, NULL, NULL);
	if (stt != CL_SUCCESS) {
		char buff[0x10000];
		clGetProgramBuildInfo(program, dv, CL_PROGRAM_BUILD_LOG, 0x10000, buff, NULL);
		fprintf(stderr, "ERROR: %s\n", buff);
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}

char *printBytes(cl_ulong bytes, char *buff) {
	unsigned int G, M, K, B;
	G = bytes / (1024 * 1024 * 1024);
	bytes %= (1024 * 1024 * 1024);
	M = bytes / (1024 * 1024);
	bytes %= (1024 * 1024);
	K = bytes / (1024);
	bytes %= (1024);
	B = bytes;
	sprintf(buff,  "%uGB %uMB %u kB %uB", G, M, K, B);
	return buff;
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
