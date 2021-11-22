
//
// Created by Xx220xX on 24/10/2020.
//

#ifndef CNN_GPU_CAMADAConvNc_H
#define CNN_GPU_CAMADAConvNc_H

#include"camada.h"


typedef struct CamadaConvNC_t {
	/// classe mãe
	Camada_t super;
	/// filtros
	Tensor W;
	/// gradiente dos filtros
	Tensor dW;
	/// soma convolucional
	Tensor z;
	/// gradiente da soma convolucional
	Tensor dz;
	/// passo
	size_t passox, passoy;
	/// largura da abertura
	size_t largx, largy;
	/// função de ativação
	uint32_t  activationFuntion;
	/// derivada da função de ativação
	uint32_t  derivationFunction;
	/// geração aleatória dos filtros
	RdP rdp_filtros;
	/***
 * @goal Fazer a soma convolucional e aplicar a função de ativação
 * @iteration  dimensão da saída s(x,y,z)
 * @param W Tensor filtros (leitura)
 * @param A Tensor de entrada (leitura)
 * @param Z Tensor soma convolucional (escrita)
 * @param S Tensor saída (escrita)
 * @param fid Identificador da função de ativação
 * @param passox passo na dimensão x
 * @param passoy passo na dimensão y
 * @param largx largura de abertura na dimensão x
 * @param largy largura de abertura na dimensão y
 * @param entradatx dimensão x da entrada
 * @param entradaty dimensão y da entrada
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão y da saída
 * @param fx dimensão x do filtro
 * @param fy dimensão y do filtro
 * @param fz dimensão z do filtro
 * @param k0 usado internamente no kernel
 */
	Kernel convncSum;
	/***
 * @goal Calcular os gradientes da soma convolucional
 * @iteration  dimensão da saída s(x,y,z)
 * @param ds Tensor gradiente da saída (leitura)
 * @param z Tensor soma convolucional(leitura)
 * @param dz Tensor gradiente da soma convolucional (escrita)
 * @param fid Identificador da função de ativação
 * @param k0 usado internamente no kernel
 */
	Kernel convncCalcGradZ;
	/***
 * @goal Calcular os gradientes dos filtros
 * @goal Atualizar os valores dos filtros
 * @iteration  dimensão dos filtros w(x,y,z,l)
 * @param dz Tensor gradiente da soma convolucional (leitura)
 * @param A Tensor de entrada (leitura)
 * @param W Tensor filtros (leitura e escrita)
 * @param dW Tensor gradiente dos filtros (leitura e escrita)
 * @param dw_x dimensão x do gradiente de filtos
 * @param dw_y dimensão y do gradiente de filtos
 * @param dw_z dimensão z do gradiente de filtos
 * @param a_x dimensão x do gradiente da entrada
 * @param a_y dimensão y do gradiente da entrada
 * @param s_x dimensão x do gradiente da saída
 * @param s_y dimensão y do gradiente da saída
 * @param passox passo na dimensão xusado internamente no kernel
 * @param passoy passo na dimensão y
 * @param largx largura de abertura na dimensão x
 * @param largy largura de abertura na dimensão y
 * @param hitlearn taxa de aprendizado
 * @param momento momento do gradiente
 * @param weightDecay taxa de decaimento
 * @param k0 usado internamente no kernel
 */
	Kernel convncCalcFiltro;
	/***
 * @goal Calcular o gradiente da entrada
 * @iteration Itera na dimensão da entrada dA(x,y,z)
 * @param W Tensor dos filtros (leitura)
 * @param DA Tensor gradiente da entrada (escrita)
 * @param dz Tensor gradiente da soma convolucional (leitura)
 * @param passox passo na dimensão x
 * @param passoy passo na dimensão y
 * @param largx largura de abertura na dimensão x
 * @param largy largura de abertura na dimensão y
 * @param entradatx dimensão x da entrada
 * @param entradaty dimensão y da entrada
 * @param saidatx dimensão x da saída
 * @param saidaty dimensão y da saída
 * @param fx dimensão x do filtro
 * @param fy dimensão y do filtro
 * @param fz dimensão z do filtro
 * @param k0 usado internamente no kernel
 */
	Kernel convncCalcGrads;

} *CamadaConvNC, CamadaConvNC_t;


extern Camada CamadaConvNC_new(Gpu gpu, Queue queue, P2d passo,P2d abertura, P3d filtro, P3d size_in, uint32_t  ativacao, Tensor entrada,
							   Parametros params, Ecx ecx, RdP rdp_filtros);


#endif //CNN_GPU_CAMADAConvNc_H
