#pragma once
#include "tensor_t.h"
#include "metodo_otimizacao.h"
#include "camada_fc_t.h"
#include "camada_pool_t.h"
#include "camada_relu_t.h"
#include "camada_conv_t.h"
#include "camada_dropout_t.h"

static void calc_grads( camada_t* camada, tensor_t<float>& grad_prox_camada )
{
	switch ( camada->tipo )
	{
		case tipo_camada::conv:
			((camada_conv_t*)camada)->calc_grads( grad_prox_camada );
			return;
		case tipo_camada::relu:
			((camada_relu_t*)camada)->calc_grads( grad_prox_camada );
			return;
		case tipo_camada::fc:
			((camada_fc_t*)camada)->calc_grads( grad_prox_camada );
			return;
		case tipo_camada::pool:
			((camada_pool_t*)camada)->calc_grads( grad_prox_camada );
			return;
		case tipo_camada::dropout_layer:
			((camada_dropout_t*)camada)->calc_grads( grad_prox_camada );
			return;
		default:
			assert( false );
	}
}

static void corrige_pesos( camada_t* camada )
{
	switch ( camada->tipo )
	{
		case tipo_camada::conv:
			((camada_conv_t*)camada)->corrige_pesos();
			return;
		case tipo_camada::relu:
			((camada_relu_t*)camada)->corrige_pesos();
			return;
		case tipo_camada::fc:
			((camada_fc_t*)camada)->corrige_pesos();
			return;
		case tipo_camada::pool:
			((camada_pool_t*)camada)->corrige_pesos();
			return;
		case tipo_camada::dropout_layer:
			((camada_dropout_t*)camada)->corrige_pesos();
			return;
		default:
			assert( false );
	}
}

static void ativa( camada_t* camada, tensor_t<float>& entrada )
{
	switch ( camada->tipo )
	{
		case tipo_camada::conv:
			((camada_conv_t*)camada)->ativa( entrada );
			return;
		case tipo_camada::relu:
			((camada_relu_t*)camada)->ativa( entrada );
			return;
		case tipo_camada::fc:
			((camada_fc_t*)camada)->ativa( entrada );
			return;
		case tipo_camada::pool:
			((camada_pool_t*)camada)->ativa( entrada );
			return;
		case tipo_camada::dropout_layer:
			((camada_dropout_t*)camada)->ativa( entrada );
			return;
		default:
			assert( false );
	}
}
