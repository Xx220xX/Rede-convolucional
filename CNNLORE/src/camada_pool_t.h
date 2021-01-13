#pragma once
#include "camada_t.h"

#pragma pack(push, 1)
struct camada_pool_t
{
	tipo_camada tipo = tipo_camada::pool;
	tensor_t<float> grads_entrada;
	tensor_t<float> entrada;
	tensor_t<float> saida;
	uint16_t passo;
	uint16_t tam_filtro;

	camada_pool_t( uint16_t passo, uint16_t tam_filtro, tdsize tam_entrada )
		:
		grads_entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		entrada( tam_entrada.x, tam_entrada.y, tam_entrada.z ),
		saida(
		(tam_entrada.x - tam_filtro) / passo + 1,
			(tam_entrada.y - tam_filtro) / passo + 1,
			tam_entrada.z
		)

		/*
		 * o tamanho da saida eh dado por S = (E - F + 2Pd)/P + 1
		 * em que:
		 * 			S = tamanho da saida
		 * 			E = tamanho da entrada
		 * 			F = tamanho do filtro
		 * 			Pd = preenchimento com zeros
		 * 			P = passo
		 *
		 * nesse codigo nao esta sendo usado preenchimento com zero --> Pd = 0
		 */

	{
		this->passo = passo;
		this->tam_filtro = tam_filtro;
		assert( (float( tam_entrada.x - tam_filtro ) / passo + 1)
				==
				((tam_entrada.x - tam_filtro) / passo + 1) );

		assert( (float( tam_entrada.y - tam_filtro ) / passo + 1)
				==
				((tam_entrada.y - tam_filtro) / passo + 1) );
	}

	// mapeia um ponto com coordenadas da saida para coordenadas da entrada
	ponto_t mapeia_saida_entrada( ponto_t saida, int z )
	{
		saida.x *= passo;
		saida.y *= passo;
		saida.z = z;
		return saida;
	}

	// vaixa de variacao das coordenadas
	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	// normaliza as coordenadas
	int normaliza_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min )
			return ceil( f );
		else
			return floor( f );
	}

	// mapeia coordenadas da entrada para a saida
	range_t mapeia_entrada_saida( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normaliza_range( (a - tam_filtro + 1) / passo, saida.tamanho.x, true ),
			normaliza_range( (b - tam_filtro + 1) / passo, saida.tamanho.y, true ),
			0,
			normaliza_range( a / passo, saida.tamanho.x, false ),
			normaliza_range( b / passo, saida.tamanho.y, false ),
			(int)saida.tamanho.z - 1,
		};
	}

	void ativa( tensor_t<float>& entrada )
	{
		this->entrada = entrada;
		ativa();
	}

	void ativa()
	{
		for ( int x = 0; x < saida.tamanho.x; x++ )
		{
			for ( int y = 0; y < saida.tamanho.y; y++ )
			{
				for ( int z = 0; z < saida.tamanho.z; z++ )
				{
					ponto_t mapeado = mapeia_saida_entrada( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float mval = -FLT_MAX;
					for ( int i = 0; i < tam_filtro; i++ )
						for ( int j = 0; j < tam_filtro; j++ )
						{
							float v = entrada( mapeado.x + i, mapeado.y + j, z );
							if ( v > mval )
								mval = v;
						}
					saida( x, y, z ) = mval;
				}
			}
		}
	}

	void corrige_pesos()
	{

	}

	void calc_grads( tensor_t<float>& grad_prox_camada )
	{
		for ( int x = 0; x < entrada.tamanho.x; x++ )
		{
			for ( int y = 0; y < entrada.tamanho.y; y++ )
			{
				range_t rn = mapeia_entrada_saida( x, y );
				for ( int z = 0; z < entrada.tamanho.z; z++ )
				{
					float soma_erro = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * passo;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * passo;

							int teste_max = entrada( x, y, z ) == saida( i, j, z ) ? 1 : 0;
							soma_erro += teste_max * grad_prox_camada( i, j, z );
						}
					}
					grads_entrada( x, y, z ) = soma_erro;
				}
			}
		}
	}
};
#pragma pack(pop)
