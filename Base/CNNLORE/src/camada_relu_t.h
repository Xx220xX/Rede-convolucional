#pragma once
#include "camada_t.h"

#pragma pack(push, 1)
struct camada_relu_t
{
	tipo_camada tipo = tipo_camada::relu;
	tensor_t<float> grads_entrada;
	tensor_t<float> entrada;
	tensor_t<float> saida;

	camada_relu_t( tdsize tam_entrada )
		:
		grads_entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		saida( tam_entrada.x, tam_entrada.y, tam_entrada.z )

	/*
	 * nas camadas ReLU a saida tem o mesmo tamanho da entrada
	 */

	{
	}


	void ativa( tensor_t<float>& entrada )
	{
		this->entrada = entrada;
		ativa();
	}

	void ativa()
	{
		for ( int i = 0; i < entrada.tamanho.x; i++ )
			for ( int j = 0; j < entrada.tamanho.y; j++ )
				for ( int z = 0; z < entrada.tamanho.z; z++ )
				{
					float v = entrada( i, j, z );
					if ( v < 0 )
						v = 0;
					saida( i, j, z ) = v;
				}

	}

	void corrige_pesos()
	{

	}

	void calc_grads( tensor_t<float>& grad_prox_camada )
	{
		for ( int i = 0; i < entrada.tamanho.x; i++ )
			for ( int j = 0; j < entrada.tamanho.y; j++ )
				for ( int z = 0; z < entrada.tamanho.z; z++ )
				{
					grads_entrada( i, j, z ) = (entrada( i, j, z ) < 0) ?
						(0) :
						(1 * grad_prox_camada( i, j, z ));
				}
	}
};
#pragma pack(pop)
