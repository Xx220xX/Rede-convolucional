//
// Created by Xx220xX on 06/05/2020.
//

#ifndef GAME2D_CONFIG_GPU_ACCESS_H
#define GAME2D_CONFIG_GPU_ACCESS_H
// funcionando
#define API_GPU_CALL(a, z, w, a_, b, act)\
//    iter_call( a.v, a.m, a.n,z.v, z.m, z.n,w.v,w.m, w.n,a_.v, a_.m, a_.n,b.v, b.m, b.n,act)

// funcionando
#define API_GPU_LAST_LAYER(dzL, aL, out, dwL, a_L_1, w, b)\
//    last_layer_learn( dzL.v,  dzL.m, aL.v, out, dwL.v, a_L_1.v,  a_L_1.m, w.v, b.v)

#define API_GPU_HIDDEN_LAYER(dzl, wl_up, dzl_up, zl, al_down, dwl, wl, bl, id_ativate_function)\
//    iter_aprende(dzl.v, dzl.m,wl_up.v, wl_up.n,dzl_up.v, dzl_up.m, dzl_up.n,zl.v, dzl.n,al_down.v, \
//                 al_down.m,dwl.v,wl.v,bl.v,id_ativate_function)
#define API_GPU_SET_WEIGHT(dzl, dwl, w, b, hit_learn)\
//    ajusta_pesos(dzl.v, dzl.m,dwl.v, dwl.m*dwl.n ,w.v,b.v, hit_learn)
#define INIT_GPU()
#define END_GPU()


#define __CL_ENABLE_EXCEPTIONS

//#include <CL/cl.hpp>
//#include <CL/CLUtil.hpp>
struct __API_CL{
	int init();
	void c_2_kernel_double_p(double *p,int len);
}API_CL;
int __API_CL::init(){
return 0;
}

void __API_CL::c_2_kernel_double_p(double *p, int len) {

}
/*
#include <cstdio>
#include <cstdlib>
#include <iostream>

const char *helloStr = "__kernel void "
					   "hello(void) "
					   "{ "
					   "  "
					   "} ";

int
maink(void) {
	cl_int err = CL_SUCCESS;
	try {

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		cl_context_properties properties[] =
				{CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
		cl::Context context(CL_DEVICE_TYPE_CPU, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		cl::Program::Sources source(1,
									std::make_pair(helloStr, strlen(helloStr)));
		cl::Program program_ = cl::Program(context, source);
		program_.build(devices);

		cl::Kernel kernel(program_, "hello", &err);

		cl::Event event;
		cl::CommandQueue queue(context, devices[0], 0, &err);
		queue.enqueueNDRangeKernel(
				kernel,
				cl::NullRange,
				cl::NDRange(4, 4),
				cl::NullRange,
				NULL,
				&event);

		event.wait();
	}
	catch (cl::Error err) {
		std::cerr
				<< "ERROR: "
				<< err.what()
				<< "("
				<< err.err()
				<< ")"
				<< std::endl;
	}

	return EXIT_SUCCESS;
}
*/
#endif //GAME2D_CONFIG_GPU_ACCESS_H
