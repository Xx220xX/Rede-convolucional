#pragma once


struct gradiente_t
{
	float grad;
	float oldgrad;
	gradiente_t()
	{
		grad = 0;
		oldgrad = 0;
	}
};
