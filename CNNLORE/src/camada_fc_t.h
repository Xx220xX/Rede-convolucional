#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "camada_t.h"

#pragma pack(push, 1)
struct camada_fc_t
{
	tipo_camada tipo = tipo_camada::fc;
	tensor_t<float> grads_entrada;
	tensor_t<float> entrada;
	tensor_t<float> saida;
	std::vector<float> input;
	tensor_t<float> pesos;
	std::vector<gradiente_t> gradientes;

	camada_fc_t( tdsize tam_entrada, int tam_saida )
		:
		grads_entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		saida( tam_saida, 1, 1 ),
		pesos( tam_entrada.x*tam_entrada.y*tam_entrada.z, tam_saida, 1 )
	{
		input = std::vector<float>( tam_saida );
		gradientes = std::vector<gradiente_t>( tam_saida );


		int valmax = tam_entrada.x * tam_entrada.y * tam_entrada.z;

		for ( int i = 0; i < tam_saida; i++ )
			for ( int h = 0; h < tam_entrada.x*tam_entrada.y*tam_entrada.z; h++ )
				pesos( h, i, 0 ) = 2.19722f / valmax * rand() / float( RAND_MAX );
		// 2.19722f = f^-1(0.9) => x em que [1 / (1 + exp(-x) ) = 0.9]
	}

	float funcao_ativacao( float x )
	{
		//return tanhf( x );
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float derivada_ativacao( float x )
	{
		//float t = tanhf( x );
		//return 1 - t * t;
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	void ativa( tensor_t<float>& entrada )
	{
		this->entrada = entrada;
		ativa();
	}

	int map( ponto_t d )
	{
		return d.z * (entrada.tamanho.x * entrada.tamanho.y) +
			d.y * (entrada.tamanho.x) +
			d.x;
	}

	void ativa()
	{
		for ( int n = 0; n < saida.tamanho.x; n++ )
		{
			float valor_entrada = 0;

			for ( int i = 0; i < entrada.tamanho.x; i++ )
				for ( int j = 0; j < entrada.tamanho.y; j++ )
					for ( int z = 0; z < entrada.tamanho.z; z++ )
					{
						int m = map( { i, j, z } );
						valor_entrada += entrada( i, j, z ) * pesos( m, n, 0 );
						printf("");
					}

			input[n] = valor_entrada;

			saida( n, 0, 0 ) = funcao_ativacao( valor_entrada );
		}
	}

	void corrige_pesos()
	{
		for ( int n = 0; n < saida.tamanho.x; n++ )
		{
			gradiente_t& grad = gradientes[n];
			for ( int i = 0; i < entrada.tamanho.x; i++ )
				for ( int j = 0; j < entrada.tamanho.y; j++ )
					for ( int z = 0; z < entrada.tamanho.z; z++ )
					{
						int m = map( { i, j, z } );
						float& w = pesos( m, n, 0 );
						w = atualiza_peso( w, grad, entrada( i, j, z ) );
					}

			atualiza_gradiente( grad );
		}
	}

	void calc_grads( tensor_t<float>& grad_prox_camada )
	{
		memset( grads_entrada.dados, 0, grads_entrada.tamanho.x *grads_entrada.tamanho.y*grads_entrada.tamanho.z * sizeof( float ) );
		for ( int n = 0; n < saida.tamanho.x; n++ )
		{
			gradiente_t& grad = gradientes[n];
			grad.grad = grad_prox_camada( n, 0, 0 ) * derivada_ativacao( input[n] );

			for ( int i = 0; i < entrada.tamanho.x; i++ )
				for ( int j = 0; j < entrada.tamanho.y; j++ )
					for ( int z = 0; z < entrada.tamanho.z; z++ )
					{
						int m = map( { i, j, z } );
						grads_entrada( i, j, z ) += grad.grad * pesos( m, n, 0 );
					}
		}
	}
};
#pragma pack(pop)
