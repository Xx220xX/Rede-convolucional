//
// Created by Xx220xX on 28/10/2020.
//

#ifndef CNN_GPU_FUNCOESDEATIVACAO_H
#define CNN_GPU_FUNCOESDEATIVACAO_H
#define FSIGMOID 0
#define FTANH 2
#define FLRELU 4
#define FLIN 6
#define FALAN 8
#define FRELU 10
#define FSOFTMAX 12
#define FLAGDIF 1

#define CHECK_F_ATIVACAO(f)((f)==FALAN)||((f)==FLIN)||((f)==FLRELU)||((f)==FRELU)||((f)==FTANH)||((f)==FSIGMOID)||((f)==FSOFTMAX)
#define F_ATIVACAO_NAME(f)f==FSIGMOID?"FSIGMOID":(f==FTANH?"FTANH":(f==FRELU?"FRELU":(f==FLIN?"FLIN":(f==FALAN?"FALAN":"INVALID"))))

#include <stdint.h>
#include <immintrin.h>

typedef __int128 FAtivacao_t;

typedef union {
	struct {
		int id;
		union {
			struct {
				float less, greater;
			};
				float epsilon;


		};
	};

	FAtivacao_t mask;
} FAtivacao;

#define FATIVACAO(fid, ...) ((FAtivacao){fid,##__VA_ARGS__}).mask
#endif //CNN_GPU_FUNCOESDEATIVACAO_H
