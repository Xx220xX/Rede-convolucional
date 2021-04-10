#pragma once
#include "tipos.h"
#include "tensor_t.h"

#pragma pack(push, 1)
struct camada_t
{
	tipo_camada tipo;
	tensor_t<float> grads_entrada;
	tensor_t<float> entrada;
	tensor_t<float> saida;
};
#pragma pack(pop)
