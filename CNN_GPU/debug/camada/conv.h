//
// Created by Henrique on 28-May-21.
//

#ifndef CNN_GPU_CONV_H
#define CNN_GPU_CONV_H
#define KERNEL_FILE "../kernels/gpu.cl"
void run(Cnn c,double *entrada){
	CnnAddConvLayer(c,1,3,2);
	CnnCall(c,entrada);
	plotTensor(c->camadas[0]->entrada,c->queue,"entrada");
	plotTensor(c->camadas[0]->saida,c->queue,"saida");
	plotTensor(((CamadaConv)c->camadas[0])->filtros,c->queue,"filtros");
	printf("%s\n",c->camadas[0]->toString(c->camadas[0]));
}
#endif //CNN_GPU_CONV_H
