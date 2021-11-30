//
// Created by Xx220xX on 12/05/2020.
//

#include "gpu/Gpu.h"
#include<stdio.h>
#include<stdlib.h>


#define check_error(error, end)if(error){char *err = Gpu_errormsg(error);fflush(stdout); fprintf(stderr,"Error %d  in file %s at line %d\n%s\n",error,__FILE__,__LINE__,err); \
fflush(stderr);free_mem(err);goto end;}

#define asprintf(str, format, ...){int len = snprintf(NULL,0,format,## __VA_ARGS__);\
str = alloc_mem(len+1,1);                                                           \
snprintf(str,len,format,## __VA_ARGS__);}

#define PAD " "
#define apendstr(str, len, format, ...) { \
         size_t sz = snprintf(NULL,0,format,##__VA_ARGS__); \
         if(!str)                         \
         str = alloc_mem(1,sz+1);    \
         else                                 \
         str = realloc(str,len+sz+1);                              \
         char *tmp = str+len;               \
         len = len+sz;\
         sprintf(tmp,format,##__VA_ARGS__) ;                           \
}

int Gpu_compileProgram(Gpu self, char *program);

int Gpu_compileProgramFile(Gpu self, char *file_program);

Queue Gpu_Queue_new(Gpu self,cl_int *error);
void Gpu_release(Gpu *self) {
	if (!self)return;
	if (!*self)return;
	if ((*self)->program)clReleaseProgram((*self)->program);
	if ((*self)->context)clReleaseContext((*self)->context);
	if ((*self)->device)clReleaseDevice((*self)->device);
	free_mem(*self);
}

CLInfo Gpu_getClinfo(Gpu cl);

cl_program compileProgram(cl_context ct, cl_device_id dv, const char *source, int *error) {
	if (!source)return NULL;
	size_t length = strlen(source);
	int _error_ = 0;
	if (!error)error = &_error_;
	cl_program program = clCreateProgramWithSource(ct, // contexto
												   1,       // numero de strings
												   &source,    // strings
												   &length, // tamanho de cada string
												   error   // error check
	);
	check_error(*error, end);
	*error = clBuildProgram(program, 1, &dv, NULL, NULL, NULL);
	if (*error != CL_SUCCESS) {
		char *buff = NULL;
		size_t len = 0;
		clGetProgramBuildInfo(program, dv, CL_PROGRAM_BUILD_LOG, 0, buff, &len);
		buff = alloc_mem(len + 1, 1);
		clGetProgramBuildInfo(program, dv, CL_PROGRAM_BUILD_LOG, len, buff, NULL);
		fprintf(stderr, "CNN_ERROR: %s\n", buff);
		clReleaseProgram(program);
		free_mem(buff);
		return NULL;
	}
	end:
	return program;
}

Gpu Gpu_new() {
	Gpu self = alloc_mem(1, sizeof(Gpu_t));

	self->type_device = CL_DEVICE_TYPE_GPU;
	// get platform
	self->error = clGetPlatformIDs(1, &self->platformId, NULL);
	check_error(self->error, metodos);
	// get device
	self->error = clGetDeviceIDs(self->platformId, self->type_device, 1, &self->device, NULL);
	check_error(self->error, metodos);
	// max compute
	self->error = clGetDeviceInfo(self->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &self->compute_units, NULL);
	check_error(self->error, metodos);
	// create context
	self->context = clCreateContext(NULL, 1, &self->device, 0, NULL, &self->error);
	check_error(self->error, metodos);
	// max work group
	size_t maxLW = 1;
	self->error = clGetDeviceInfo(self->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLW, NULL);
	check_error(self->error, metodos);
	self->maxworks = maxLW;

	// metodos
	metodos:
	self->release =Gpu_release;
	self->errorMsg = Gpu_errormsg;
	self->compileProgram =  Gpu_compileProgram;
	self->compileProgramFile =  Gpu_compileProgramFile;
	self->getClInfo =  Gpu_getClinfo;
	self->Queue_new =  Gpu_Queue_new;

	return self;
}

#define allocprint(dst, format, ...) {int len = snprintf(NULL,0,format,##__VA_ARGS__); dst = alloc_mem(1,len+1);snprintf(dst,len+1,format,##__VA_ARGS__);}

char *Gpu_errormsg(int error) {
	char *msg;
	switch (error) {
		case 0: allocprint(msg, "%s", "SUCCESS");
			break;
		case -1: allocprint(msg, "%s", "DEVICE_NOT_FOUND");
			break;
		case -2: allocprint(msg, "%s", "DEVICE_NOT_AVAILABLE");
			break;
		case -3: allocprint(msg, "%s", "COMPILER_NOT_AVAILABLE");
			break;
		case -4: allocprint(msg, "%s", "MEM_OBJECT_ALLOCATION_FAILURE");
			break;
		case -5: allocprint(msg, "%s", "OUT_OF_RESOURCES");
			break;
		case -6: allocprint(msg, "%s", "OUT_OF_HOST_MEMORY");
			break;
		case -7: allocprint(msg, "%s", "PROFILING_INFO_NOT_AVAILABLE");
			break;
		case -8: allocprint(msg, "%s", "MEM_COPY_OVERLAP");
			break;
		case -9: allocprint(msg, "%s", "IMAGE_FORMAT_MISMATCH");
			break;
		case -10: allocprint(msg, "%s", "IMAGE_FORMAT_NOT_SUPPORTED");
			break;
		case -11: allocprint(msg, "%s", "BUILD_PROGRAM_FAILURE");
			break;
		case -12: allocprint(msg, "%s", "MAP_FAILURE");
			break;
		case -13: allocprint(msg, "%s", "MISALIGNED_SUB_BUFFER_OFFSET");
			break;
		case -14: allocprint(msg, "%s", "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
			break;
		case -15: allocprint(msg, "%s", "COMPILE_PROGRAM_FAILURE");
			break;
		case -16: allocprint(msg, "%s", "LINKER_NOT_AVAILABLE");
			break;
		case -17: allocprint(msg, "%s", "LINK_PROGRAM_FAILURE");
			break;
		case -18: allocprint(msg, "%s", "DEVICE_PARTITION_FAILED");
			break;
		case -19: allocprint(msg, "%s", "KERNEL_ARG_INFO_NOT_AVAILABLE");
			break;
		case -30: allocprint(msg, "%s", "INVALID_VALUE");
			break;
		case -31: allocprint(msg, "%s", "INVALID_DEVICE_TYPE");
			break;
		case -32: allocprint(msg, "%s", "INVALID_PLATFORM");
			break;
		case -33: allocprint(msg, "%s", "INVALID_DEVICE");
			break;
		case -34: allocprint(msg, "%s", "INVALID_CONTEXT");
			break;
		case -35: allocprint(msg, "%s", "INVALID_QUEUE_PROPERTIES");
			break;
		case -36: allocprint(msg, "%s", "INVALID_COMMAND_QUEUE");
			break;
		case -37: allocprint(msg, "%s", "INVALID_HOST_PTR");
			break;
		case -38: allocprint(msg, "%s", "INVALID_MEM_OBJECT");
			break;
		case -39: allocprint(msg, "%s", "INVALID_IMAGE_FORMAT_DESCRIPTOR");
			break;
		case -40: allocprint(msg, "%s", "INVALID_IMAGE_SIZE");
			break;
		case -41: allocprint(msg, "%s", "INVALID_SAMPLER");
			break;
		case -42: allocprint(msg, "%s", "INVALID_BINARY");
			break;
		case -43: allocprint(msg, "%s", "INVALID_BUILD_OPTIONS");
			break;
		case -44: allocprint(msg, "%s", "INVALID_PROGRAM");
			break;
		case -45: allocprint(msg, "%s", "INVALID_PROGRAM_EXECUTABLE");
			break;
		case -46: allocprint(msg, "%s", "INVALID_KERNEL_NAME");
			break;
		case -47: allocprint(msg, "%s", "INVALID_KERNEL_DEFINITION");
			break;
		case -48: allocprint(msg, "%s", "INVALID_KERNEL");
			break;
		case -49: allocprint(msg, "%s", "INVALID_ARG_INDEX");
			break;
		case -50: allocprint(msg, "%s", "INVALID_ARG_VALUE");
			break;
		case -51: allocprint(msg, "%s", "INVALID_ARG_SIZE");
			break;
		case -52: allocprint(msg, "%s", "INVALID_KERNEL_ARGS");
			break;
		case -53: allocprint(msg, "%s", "INVALID_WORK_DIMENSION");
			break;
		case -54: allocprint(msg, "%s", "INVALID_WORK_GROUP_SIZE");
			break;
		case -55: allocprint(msg, "%s", "INVALID_WORK_ITEM_SIZE");
			break;
		case -56: allocprint(msg, "%s", "INVALID_GLOBAL_OFFSET");
			break;
		case -57: allocprint(msg, "%s", "INVALID_EVENT_WAIT_LIST");
			break;
		case -58: allocprint(msg, "%s", "INVALID_EVENT");
			break;
		case -59: allocprint(msg, "%s", "INVALID_OPERATION");
			break;
		case -60: allocprint(msg, "%s", "INVALID_GL_OBJECT");
			break;
		case -61: allocprint(msg, "%s", "INVALID_BUFFER_SIZE");
			break;
		case -62: allocprint(msg, "%s", "INVALID_MIP_LEVEL");
			break;
		case -63: allocprint(msg, "%s", "INVALID_GLOBAL_WORK_SIZE");
			break;
		case -64: allocprint(msg, "%s", "INVALID_PROPERTY");
			break;
		case -65: allocprint(msg, "%s", "INVALID_IMAGE_DESCRIPTOR");
			break;
		case -66: allocprint(msg, "%s", "INVALID_COMPILER_OPTIONS");
			break;
		case -67: allocprint(msg, "%s", "INVALID_LINKER_OPTIONS");
			break;
		case -68: allocprint(msg, "%s", "INVALID_DEVICE_PARTITION_COUNT");
			break;
		default:
			break;
	}
	return msg;
}

int Gpu_compileProgram(Gpu self, char *program) {
	if (self->error)goto end;
	// compile program kernel
	self->program = compileProgram(self->context, self->device, program, &self->error);
	check_error(self->error, end);
	end:
	return self->error;
}

int Gpu_compileProgramFile(Gpu self, char *file_program) {
	if (self->error)return self->error;

	char *program = NULL;
	FILE *f;
	f = fopen(file_program, "r");
	if (!f) {
		fprintf(stderr, "arquivo nao encontrado no caminho %s\n", file_program);
		self->error = 1;
		return 1;
	}
	long int size = 0;
	fseek(f, 0L, SEEK_END);
	size = ftell(f);
	program = alloc_mem(size + 1, sizeof(char));
	fseek(f, 0L, SEEK_SET);
	fread(program, sizeof(char), size, f);
	program[size] = 0;
	fclose(f);
	int error = self->compileProgram(self, program);
	free_mem(program);
	return error;
}


char *CLInfo_json(CLInfo *self) {
	char *json;
	int len = 0;
	apendstr(json, len, "{\n"
			PAD"\"device_name\":\"%s\",\n"
			PAD"\"hardware_version\":\"%s\",\n"
			PAD"\"software_version\":\"%s\",\n"
			PAD"\"openCL_version\":\"%s\",\n"
			PAD"\"global_mem_cache_size\":%llu,\n"
			PAD"\"global_mem_cache_line_size\":%llu,\n"
			PAD "\"global_mem_size\":%llu,\n"
			PAD"\"local_mem_size\":%llu,\n"
			PAD"\"max_freq_mhz\":%u,\n"
			PAD"\"max_mem_alloc_size\":%llu,\n"
			PAD"\"max_work_group_size\":%llu,\n"
			PAD"\"max_work_item_sizes\":[%zu,%zu,%zu]\n"
			   "}", self->device_name, self->hardware_version, self->software_version, self->openCL_version, self->global_mem_cache_size, self->global_mem_cache_line_size, self->global_mem_size, self->local_mem_size, self->max_freq_mhz, self->max_mem_alloc_size, self->max_work_group_size, self->max_work_item_sizes[0], self->max_work_item_sizes[1], self->max_work_item_sizes[2]
	);
	return json;
}

CLInfo Gpu_getClinfo(Gpu cl) {
	CLInfo cif = {0};
	clGetDeviceInfo(cl->device, CL_DEVICE_NAME, 64, cif.device_name, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_VERSION, 64, cif.hardware_version, NULL);
	clGetDeviceInfo(cl->device, CL_DRIVER_VERSION, 64, cif.software_version, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_OPENCL_C_VERSION, 64, cif.openCL_version, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cif.global_mem_cache_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_ulong), &cif.global_mem_cache_line_size, NULL);
	clGetDeviceInfo(cl->device, CL_MEM_SIZE, sizeof(cl_ulong), &cif.global_mem_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &cif.local_mem_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &cif.max_freq_mhz, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &cif.max_mem_alloc_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cif.max_work_group_size, NULL);
	clGetDeviceInfo(cl->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &cif.max_work_item_sizes, NULL);
	cif.json = (char *(*)(void *)) CLInfo_json;
	return cif;
}

Queue Gpu_Queue_new(Gpu self,cl_int *error){
	return clCreateCommandQueueWithProperties(self->context,self->device,NULL,error);
}

