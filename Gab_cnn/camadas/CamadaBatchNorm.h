//
// Created by Henrique on 8/5/2021.
//

#ifndef CNN_GPU_CAMADABATCHNORM_H
#define CNN_GPU_CAMADABATCHNORM_H

#include "camada.h"
#include <stdlib.h>


typedef struct CamadaBatchNorm_t {
	/// classe mãe Camada_t
	Camada_t super;
	/// Gama
	Tensor Y;
	/// gradiente de Gama
	Tensor gradY;
	/// BETA
	Tensor B;
	/// grandiente de BETA
	Tensor gradB;
	/// armazena a media do tensor de entrada
	Tensor media;
	/// armazena a soma do tensor diferença
	Tensor somaDiferenca;
	/// armazena a variancia
	Tensor variancia;
	/// armazena a derivada parcial em relação a variância
	Tensor gradVariancia;
	/// armazena a entrada menos a média
	Tensor diferenca;
	/// armazena entrada menos a média ao quadrado
	Tensor diferencaquad;
	/// armazena a entrada normalizada
	Tensor norma;
	/// constante para evitar divisão por zero
	REAL epsilon;
	/***
 * Itera na dimensão z da entrada
 * possou alto custo pois faz a media das dimensoes (:,:,z)
 * @param entrada Tensor de entrada (leitura)
 * @param media Tensor para media das dimensoes (escrita) (1,1,z)
 * @param entradatx dimensão x da entrada
 * @param entradaty dimensão y da entrada
 * @param k0 usado internamente no kernel
 */
	Kernel batchNormAtiva1;// calcula a media
	/**
 * Itera na entrada (x,y,z)
 * @param entrada Tensor entrada (leitura)
 * @param media Tensor media (leitura)
 * @param diferenca Tensor diferença (mesmo tamanho da entrada)(escrita)
 * @param diferencaquad tensor diferença quadratica(escrita)
 * @param entradatx dimensção x da entrada
 * @param entradaty dimensão y da entada
 * @param k0 usada internamente no kernel
 */
	Kernel batchNormAtiva2;// calcula a diferenca
	/***
 * Itera na dimensão z da entrada
 * @param dif tensor de diferenca (leitura)
 * @param difQuad tensor de diferença quadrática (leitura)
 * @param sumdiferenca tensor para soma de diferenças (escrita)
 * @param variancia tensor para variância (escrita)
 * @param episolon valor para evitar divisão por zero
 * @param diftx dimensão x da diferença
 * @param difty dimensão y da diferença
 * @param k0  usado internamente pelo kernel
 */
	Kernel batchNormAtiva3;// calcula a variancia
	/***
 * itera na dimensão da entrada (x,z,y)
 * Faz a normalização
 * Y = (x-media)/variancia* Y + B
 * @param saida Tensor para a saída (escrita)
 * @param norma Tensor para os valores normalizados (escrita)
 * @param diferenca Tensor da diferença (leitura)
 * @param variancia Tensor da variância (leitura)
 * @param Y Tensor Gama (leitura)
 * @param B Tensor Beta (leitura)
 * @param diferencatx dimensão x da diferença
 * @param diferencaty dimensão da diferença
 * @param k0 usado internamente no kernel
 */
	Kernel batchNormAtiva4;// normaliza
	/**
 * Itera na dimensão da entrada
 * Calcula os gradientes de entrada
 * @param gradIn Tensor gradiente de entrada (escrite e leitura)
 * @param gradNext Tensor gradiente de saída (leitura)
 * @param variancia Tensor variância (leitura)
 * @param media Tensor média (leitura)
 * @param Y Tensor Gama (leitura)
 * @param somaDif Tensor somadif (leitura)
 * @param entrada Tensor entrada (leitura)
 * @param entradatx dimensão x da entrada
 * @param entradaty dimensão y da entrada
 * @param k0 usado internamente no kernel
 */
	Kernel batchNormCalcGrads1;// calcula gradientes de entrada
	/***
 * Itera na dimensão z da entrada
 *
 *
 * Calcula o gradiente de Y e B e realiza o aprendizado
 * @param gradNext Tensor gradiente da saída(leitura)
 * @param norma Tensor normalizado (leitura)
 * @param Y Tensor Gama (leitura e escrita)
 * @param B Tensor Beta (leitura e escrita)
 * @param gradY Tensor gradiente de Gama (leitura e escrita)
 * @param gradB Tensor gradiente de Beta (leitura e escrita)
 * @param hitlearn taxa de aprendizado
 * @param momento taxa de momento
 * @param weightDecay taxa de decaimento
 * @param entradatx dimensão x da entrada
 * @param entradaty dimensão y da entrad
 * @param k0 usado internamente no kernel
 */
	Kernel batchNormCalcGrads2;//calcula gradiente Y B
	/// Parametros de geração aleatória do tensor Gama
	RdP rdp_Y;
	/// Parametros de geração aleatória do tensor Beta
	RdP rdp_B;
} *CamadaBatchNorm, CamadaBatchNorm_t;

extern Camada CamadaBatchNorm_new(Gpu gpu, Queue queue, Parametros params, P3d size_in, Tensor entrada, REAL epsilon, Ecx ecx, RdP randY, RdP randB);


#endif //CNN_GPU_CAMADABATCHNORM_H
