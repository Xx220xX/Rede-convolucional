//
// Created by Xx220xX on 12/05/2020.
//

#include "WrapperCL.h"

int WrapperCL_init(WrapperCL *self, const char *src) {
    const size_t length = strlen(src);
    cl_int error = CL_SUCCESS;

    // getPlatform
    error = clGetPlatformIDs(1, &self->platformId, NULL);
    PERR(error,"falha ao pegar plataformas");

    // get device
    error = clGetDeviceIDs(self->platformId, CL_DEVICE_TYPE_GPU, 1, &self->device, NULL);
    PERR(error,"falha ao pegar dispositivos");

    // get size of compute
    error = clGetDeviceInfo(self->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint), &self->compute_units, NULL);
    PERR(error,"falha ao pegar informacao do dispositivo");

    // create context
    self->context = clCreateContext(NULL, 1, &self->device, 0, NULL, &error);
    PERR(error,"failed when try create context");

    // compile program kernel
    self->program = clCreateProgramWithSource(self->context, // contexto
                                              1,       // numero de strings
                                              &src,    // strings
                                              &length, // tamanho de cada string
                                              &error   // error check
    );
    PERR(error,"failed to create program");

    // build
    cl_int stt = clBuildProgram(self->program, 1, &self->device, NULL, NULL, NULL);

    if (stt != CL_SUCCESS) {
        char buff[0x10000];
        clGetProgramBuildInfo(self->program, self->device, CL_PROGRAM_BUILD_LOG, 0x10000, buff, NULL);
        fprintf(stderr, "ERROR: %s\n", buff);

        return stt;
    }

    return 0;

}

void WrapperCL_release(WrapperCL *self) {
    clReleaseProgram(self->program);
    clReleaseDevice(self->device);
    clReleaseContext(self->context);
}
void showError(int error) {
    switch (error) {
        case CL_BUILD_PROGRAM_FAILURE:
            fprintf(stderr, "%s\n", "BUILD_PROGRAM_FAILURE");
            break;
        case CL_DEVICE_NOT_FOUND:
            fprintf(stderr, "%s\n", "DEVICE_NOT_FOUND");
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            fprintf(stderr, "%s\n", "COMPILER_NOT_AVAILABLE");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            fprintf(stderr, "%s\n", "MEM_OBJECT_ALLOCATION_FAILURE");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            fprintf(stderr, "%s\n", "OUT_OF_HOST_MEMORY");
            break;
        case CL_OUT_OF_RESOURCES:
            fprintf(stderr, "%s\n", "OUT_OF_RESOURCES");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            fprintf(stderr, "%s\n", "PROFILING_INFO_NOT_AVAILABLE");
            break;
        case CL_MEM_COPY_OVERLAP:
            fprintf(stderr, "%s\n", "MEM_COPY_OVERLAP");
            break;
        case CL_MAP_FAILURE:
            fprintf(stderr, "%s\n", "MAP_FAILURE");
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            fprintf(stderr, "%s\n", "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
            break;
        case CL_LINK_PROGRAM_FAILURE:
            fprintf(stderr, "%s\n", "LINK_PROGRAM_FAILURE");
            break;
        case CL_LINKER_NOT_AVAILABLE:
            fprintf(stderr, "%s\n", "LINKER_NOT_AVAILABLE");
            break;
        case CL_DEVICE_PARTITION_FAILED:
            fprintf(stderr, "%s\n", "DEVICE_PARTITION_FAILED");
            break;
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            fprintf(stderr, "%s\n", "KERNEL_ARG_INFO_NOT_AVAILABLE");
            break;

        default:
            fprintf(stderr, "UNKNOW error code %d\n", error);
    }
    fprintf(stderr, "\n");


}
