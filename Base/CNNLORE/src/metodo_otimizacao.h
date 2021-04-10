#pragma once
#include "gradiente_t.h"

#define TAXA_APRENDIZADO 0.01
#define MOMENTO 0.6
#define WEIGHT_DECAY 0.001

static float atualiza_peso( float w, gradiente_t& grad, float multp = 1 )
{
	float m = (grad.grad + grad.oldgrad * MOMENTO);
	w -= TAXA_APRENDIZADO  * m * multp +
		 TAXA_APRENDIZADO * WEIGHT_DECAY * w;
	return w;
}

static void atualiza_gradiente( gradiente_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTO);
}
