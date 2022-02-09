//
// Created by Henrique on 30/11/2021.
//
#include "cnn/cnn.h"

#include "camadas/CamadaPool.h"
#include "camadas/all_camadas.h"
#include<string.h>

int main() {
	Cnn c = Cnn_new();
	c->setInput(c, 3, 3, 2);
	c->Pooling(c, P2D(1, 1), P2D(2, 2), MAXPOOL);

	REAL in[] = { 0.6947684528,0.9201158278, 0.1838218198,
				 0.5719285763, 0.6964326911, 0.746596011,
				 0.6688675721, 0.249482706, 0.04509204255,

				 0.6974112558, 0.7588798661, 0.995288516,
				 0.566216851, 0.3159716933, 0.4708597018,
				 0.520923498, 0.9162451593, 0.1948300399
	};
	REAL ds[] = {0.5876862434, 0.8260275344,
				 0.771309978, 0.5041622867,

				 0.7703319531, 0.8215513627,
				 0.4167639388, 0.9404333207};
	c->predictv(c, in);
	c->ds->setvalues(c->ds,ds);
	CamadaPool  cp = CST_POOL(c->cm[0]);
	cp->super.da = Tensor_new(3,3,2,1,c->ecx,0,c->gpu->context,c->queue);
	cp->super.da->fill(cp->super.da,0);
	cp->super.retroPropagationBatch(cp,c->ds,1);
	cp->super.da->print(cp->super.da);
//	c->cm[0]->s->print(c->cm[0]->s);
	return c->release(&c);
}