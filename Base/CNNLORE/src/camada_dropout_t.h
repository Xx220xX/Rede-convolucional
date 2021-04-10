#pragma once
#include "camada_t.h"

#pragma pack(push, 1)
struct camada_dropout_t
{
	tipo_camada tipo = tipo_camada::dropout_layer;
	tensor_t<float> grads_entrada;
	tensor_t<float> entrada;
	tensor_t<float> saida;
	tensor_t<bool> hitmap;
	float p_ativacao;

	camada_dropout_t( tdsize tam_entrada, float p_ativacao )
		:
		grads_entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		saida( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		hitmap( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		p_ativacao( p_ativacao )
	{
	}

	void ativa( tensor_t<float>& entrada )
	{
		this->entrada = entrada;
		ativa();
	}

	void ativa()
	{
		for ( int i = 0; i < entrada.tamanho.x*entrada.tamanho.y*entrada.tamanho.z; i++ )
		{
			bool teste_ativa = (rand() % RAND_MAX) / float( RAND_MAX ) <= p_ativacao;
			hitmap.dados[i] = teste_ativa;
			saida.dados[i] = teste_ativa ? entrada.dados[i] : 0.0f;
		}
	}


	void corrige_pesos()
	{
		
	}

	void calc_grads( tensor_t<float>& grad_prox_camada )
	{
		for ( int i = 0; i < entrada.tamanho.x*entrada.tamanho.y*entrada.tamanho.z; i++ )
			grads_entrada.dados[i] = hitmap.dados[i] ? grad_prox_camada.dados[i] : 0.0f;
	}
};
#pragma pack(pop)
