//
// Created by hslhe on 14/11/2021.
//

#include <stdio.h>
#include "tensor/exc.h"

int main() {
	Ecx ecx = Ecx_new(18);
	ecx->addstack(ecx, "main");
	ecx->addstack(ecx, "func");
	ecx->error = 1;
	ecx->addstack(ecx, "func2");
	ecx->print(ecx);

	ecx->popstack(ecx);
	ecx->popstack(ecx);
	ecx->popstack(ecx);

	ecx->release(&ecx);
	return 0;
}