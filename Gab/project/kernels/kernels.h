//
// Created by Henrique on 01/01/2022.
//
#ifndef GAB_KERNEL_H
#define GAB_KERNEL_H
#include <gpu/gpu_macros.h>
#define __kernel
#define __global
// cl:utils.h

// cl:bathnorm.h
kV BatchNormMedia(Vr a, Vw u, int ax, int ay, int id_0) ;
kV BatchNormInvDesv(Vr a, Vr u, Vr o, REAL episolon, int ax, int ay, int id_0) ;
kV BatchNormNormaliza(Vw s, Vrw v, Vr a, Vr u, Vr o, Vr Y, Vr B, int ax, int ay, int id_0) ;
kV BatchNormaCalcDnorm(Vw dv, Vr ds,Vr Y, int ax, int ay, int id_0) ;
kV BatchNormMediadnorm_norma(Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, int ax, int ay, int id_0) ;
kV BatchNormaCalcDa(Vr da, Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, Vr o, int ax, int ay, int id_0) ;
kV BatchNormaCalcdYdB(Vr ds, Vr v, Vw dY, Vw dB, long batchSize, int ax, int ay, int id_0) ;
kV BatchNormaLearn(Vrw Y, Vrw B, Vrw dY, Vrw dB, REAL hit, REAL momento, REAL decaimento, int id_0) ;
// cl:cnnutils.h
kV createImg(__global unsigned char *out, Vr v, int vx, int vy, int imi, int imy, int k0) ;
kV putIMG(__global unsigned char *imagem_saida, Vr v, int z, REAL px, REAL py, int imy, int width, int i0, int j0, int vx, int vy, int k0) ;
kV normalizeVector(Vr input, Vr saida, REAL multiplicador, REAL somador, REAL subtrator, int k0) ;
kV kernel_sub(Vr ds, Vr s, Vr t, int k0) ;
kV kernel_normalizechar2real(Vr dst, __global unsigned char *src, REAL a, REAL b, int k0) ;
kV kernel_getVetorClassFromChar(Vr dst, __global unsigned char *ints, unsigned int noptiobs, int k0) ;
kV kernel_fixW(Vr w, Vr dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0) ;
// cl:conv.h
kV convSum(Vr filtro, Vr entrada, Vr saida, int passox, int passoy, int saidatx, int saidaty, int entradatx, int entradaty, int fx, int fy, int fz, int k0) ;
kV convCalcGradAndFixWeight(Vr filtros, Vr ds, Vr entrada, Vr gradFiltro, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, REAL hitLearn, REAL momento, REAL weightDecay, int k0) ;
kV convCalcGradIn(Vr filtro, Vr gradEntrada, Vr gradNext, int fx, int fy, int fz, int passox, int passoy, int entradatx, int entradaty, int saidatx, int saidaty, int saidatz, int k0) ;
kV convCalcGradBatch(Vr ds, Vr A, Vr dW, long batchSize, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, int k0) ;
// cl:convNc.h
kV convncSum(Vr W, Vr A, Vr Z, Vr S, unsigned int fid, unsigned int passox, int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0) ;
kV convncCalcGradZ(Vr ds, Vr z, Vr dz, unsigned int fid, int k0) ;
kV convncCalcGrads(Vr W, Vr DA, Vr dz, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0) ;
kV convncCalcFiltro(Vr dz, Vr A, Vr W, Vr dW, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, REAL hitlearn, REAL momento, REAL weightDecay, int k0) ;
kV convncCalcFiltroBatch(Vr dz, Vr A, Vr dW, long batchSize, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, int k0) ;
// cl:dropout.h
kV dropativa(Vr entrada, Vr saida, __global char *hitmap, long seed, REAL pativa, int k0) ;
kV dropcalcgrad(Vr gradentrada, __global char *hitmap, Vr gradnext, int k0) ;
// cl:padding.h
kV paddingfeed(Vr in, Vr out, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) ;
kV paddingBack(Vr gradNext, Vr gradin, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0) ;
// cl:poolav.h
kV poolAVativa(Vr entrada, Vr saida, int passox, int passoy, int fx, int fy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) ;
kV poolAvCalcGrads(Vr A, Vw dA, Vr dS, Vr S, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) ;
// cl:poolMax.h
kV poolativa(Vr entrada, Vr saida, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) ;
kV poolCalcGrads(Vr A, Vr dA, Vr dS, Vr S, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0) ;
// cl:poolMin.h
kV poolativaMin(Vr A, Vr S, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0) ;
// cl:prelu.h
kV preluativa(Vr A, Vw S, Vr W, int k0) ;
kV prelucalcgrad(Vw dA, Vr A, Vr dS, Vrw W, Vrw dW, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0) ;
kV preluonlyfix(Vr A, Vr dS, Vrw W, Vrw dW, REAL hitlearn, REAL momento, REAL decaimento, int k0) ;
kV prelucalcgradBatch(Vw dA, Vr A, Vr dS, Vr W, Vrw dW, long batchSize, int k0) ;
kV preluonlyDABatch(Vr A, Vr dS, Vr W, Vr dW, long batchSize, int k0) ;
// cl:relu.h
kV reluativa(Vr A, Vr S, REAL menor, REAL maior, int k0) ;
kV relucalcgrad(Vr dA, Vr A, Vr dS, REAL menor, REAL maior, int k0) ;
// cl:softmax.h
kV softmaxExp(Vr entrada, Vr exponent, int k0) ;
kV softmaxSomaExp(Vr eps, Vr soma, int saidatx, int saidaty, int k0) ;
kV softmaxNormaliza(Vr exponet, Vr soma, Vr saida, int saidatx, int saidaty, int k0) ;
kV softMaxcalcgrad(Vr da, Vr s, Vr ds, int sx, int sy, int k0) ;
kV softmaxFindMax(Vr a, Vr mx, __global int *i_max, int ax, int ay, int k0) ;
kV softmaxExpNorm(Vr entrada, Vr exponent, Vr mx, int ax, int ay, int k0) ;
#endif // GAB_KERNEL_H
