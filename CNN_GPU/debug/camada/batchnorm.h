//
// Created by Henrique on 28-May-21.
//

#ifndef CNN_GPU_CONV_H
#define CNN_GPU_CONV_H
#define KERNEL_FILE "../kernels/gpu.cl"
void run(Cnn c,double *entrada,double *saida){
	CnnAddBatchNorm(c,1e-10);
	CnnAddFullConnectLayer(c,4,FSIGMOID);
	CnnAddBatchNorm(c,1e-10);
	CnnAddFullConnectLayer(c,1,FSIGMOID);
	Camada  cm = c->camadas[c->size-1];
	c->flags = CNN_FLAG_CALCULE_ERROR;
	CnnCall(c,entrada);
	printf("aquii poha %g %g\n",entrada[0],entrada[1]);
	plotTensor(c->camadas[0]->entrada,c->queue,"entrada");


	CnnLearn(c,saida);
	plotTensor(cm->saida,c->queue,"saida1");
	for(int i=0;i<100;i++) {
		CnnCall(c, entrada);
		CnnLearn(c, saida);
		printf("%g\n",c->normaErro);
	}
	plotTensor(cm->saida,c->queue,"saida2");

	plotTensor(c->camadas[1]->entrada,c->queue,"notnorm");
	plotTensor(c->camadas[1]->saida,c->queue,"norm");
	for (int i=0;i<c->size;printf("%s\n",c->camadas[i]->toString(c->camadas[i++])));

}
#endif //CNN_GPU_CONV_H
