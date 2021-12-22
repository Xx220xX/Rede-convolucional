//
// Created by Xx220xX on 28/10/2020.
//

#ifndef CNN_GPU_FUNCOESDEATIVACAO_H
#define CNN_GPU_FUNCOESDEATIVACAO_H
#define FSIGMOID 0
#define FTANH 2
#define FRELU 4
#define FLIN 6
#define FALAN 8
#define FLAGDIF 1

#define CHECK_F_ATIVACAO(f)((f)==FALAN)||((f)==FLIN)||((f)==FRELU)||((f)==FTANH)||((f)==FSIGMOID)

#endif //CNN_GPU_FUNCOESDEATIVACAO_H
