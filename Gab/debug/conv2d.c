//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"
#include "matlab.h"
#include<string.h>
#include "camadas/CamadaConv2D.h"

int main() {
	// criar cnn
	Cnn cnn = Cnn_new();
	//criar entrada
	Tensor entrada = Tensor_new(32, 32, 3, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	cnn->setInput(cnn, entrada->x, entrada->y, entrada->z);
	// adicionar camada
	cnn->Convolucao2D(cnn, P2D(2, 2), P3D(4, 4, 2), FRELU, Params(1e-3), RDP(0));
	cnn->cm[0]->da = Tensor_new(entrada->x, entrada->y, entrada->z, 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	// criar saída
	P3d sizeout = cnn->getSizeOut(cnn);
	Tensor target = Tensor_new(unP3D(sizeout), 1, cnn->ecx, 0, cnn->gpu->context, cnn->queue);
	entrada->randomize(entrada, TENSOR_GAUSSIAN, 1, 0);
	target->randomize(target, TENSOR_GAUSSIAN, 1, 0);
	// capturar camada
	CamadaConv2D cf = (CamadaConv2D) cnn->cm[0];


	matlabInit();
	cnn->predict(cnn, entrada);
	Trmatlab(cf->super.a, "A");
	Trmatlab(cf->super.s, "S");
	Trmatlab(cf->W, "W");
	Trmatlab(cf->z, "Z");
	cnn->learnBatch(cnn, target, 1);
	Trmatlab(cnn->ds, "dS");
	Trmatlab(cf->dz, "dZ");
	Trmatlab(cf->dW, "dW");
	Trmatlab(cf->super.da, "dA");

	Trmatlab(target, "T");
	matlabf("px = %zu;\npy = %zu;\n", cf->passox, cf->passoy);
	matlab("%% matlab code");
	matlab("%calcula convolução 2d");
	matlab("Z_m = S*0;\n"
		   "for z =1:size(A,3)\n"
		   "  for l = 1:size(W,3)\n"
		   "    for x = 1:size(Z,1)\n"
		   "      for y = 1:size(Z,2)\n"
		   "        Z_m(x,y,(l-1)*size(A,3)+z) = sum((A(1+(x-1)*px:(x-1)*px+size(W,1),(y-1)*py+1:(y-1)*py+size(W,2),z).* W(:,:,l))(:)); \n"
		   "      end\n"
		   "    end\n"
		   "  end\n"
		   "end");

	matlab("%ativa");
	matlabAtivation("S_m", "Z_m", cf->fid);
	matlab("%calcula o ultimo gradiente");
	matlab("dS_m = S - T;");
	matlab("%calcula o gradiente da ativação");
	matlabAtivation("dZ_m", "Z_m", cf->dfid);
	matlab("dZ_m = dZ_m .* dS_m;");
	matlab("%calcula o gradiente de entrada");

	matlab("dA_m = A*0;\n"
		   "for z =1:size(A,3)\n"
		   "  for l = 1:size(W,3)\n"
		   "    for x = 1:size(Z,1)\n"
		   "      for y = 1:size(Z,2)\n"
		   "      dA_m(1+(x-1)*px:(x-1)*px+size(W,1),(y-1)*py+1:(y-1)*py+size(W,2),z)  = dA_m(1+(x-1)*px:(x-1)*px+size(W,1),(y-1)*py+1:(y-1)*py+size(W,2),z) + W(:,:,l) .* dZ_m(x,y,(l-1)*size(A,3)+z);   \n"
		   "      end\n"
		   "    end\n"
		   "  end\n"
		   "end");
	matlab("%calcula o gradiente dos pesos");
	matlab("dW_m = W*0;\n"
		   "for z =1:size(A,3)\n"
		   "  for l = 1:size(W,3)\n"
		   "    for x = 1:size(Z,1)\n"
		   "      for y = 1:size(Z,2)\n"
		   "      dW_m(:,:,l) = dW_m(:,:,l) + A(1+(x-1)*px:(x-1)*px+size(W,1),(y-1)*py+1:(y-1)*py+size(W,2),z) *dZ_m(x,y,(l-1)*size(A,3)+z);\n"
		   "      end\n"
		   "    end\n"
		   "  end\n"
		   "end");


	matlabCmp("dW","dW_m");
	matlabCmp("dA","dA_m");
	matlabCmp("dZ","dZ_m");
	matlabCmp("dS","dS_m");
	matlabCmp("S","S_m");
	matlabCmp("Z","Z_m");
	Release(entrada);
	Release(target);
	matlabEnd();

	return cnn->release(&cnn);

}