//utils.h

//bathnorm.h
#define Knew_BatchNormMedia(x) KRN_new(x, "BatchNormMedia","Vr a, Vw u, int ax, int ay, int id_0")
#define Knew_BatchNormInvDesv(x) KRN_new(x, "BatchNormInvDesv","Vr a, Vr u, Vr o, REAL episolon, int ax, int ay, int id_0")
#define Knew_BatchNormNormaliza(x) KRN_new(x, "BatchNormNormaliza","Vw s, Vrw v, Vr a, Vr u, Vr o, Vr Y, Vr B, int ax, int ay, int id_0")
#define Knew_BatchNormaCalcDnorm(x) KRN_new(x, "BatchNormaCalcDnorm","Vw dv, Vr ds,Vr Y, int ax, int ay, int id_0")
#define Knew_BatchNormMediadnorm_norma(x) KRN_new(x, "BatchNormMediadnorm_norma","Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, int ax, int ay, int id_0")
#define Knew_BatchNormaCalcDa(x) KRN_new(x, "BatchNormaCalcDa","Vr da, Vr v, Vr dv, Vr mdnorm, Vr mdnormnorm, Vr o, int ax, int ay, int id_0")
#define Knew_BatchNormaCalcdYdB(x) KRN_new(x, "BatchNormaCalcdYdB","Vr ds, Vr v, Vw dY, Vw dB, long batchSize, int ax, int ay, int id_0")
#define Knew_BatchNormaLearn(x) KRN_new(x, "BatchNormaLearn","Vrw Y, Vrw B, Vrw dY, Vrw dB, REAL hit, REAL momento, REAL decaimento, int id_0")

//cnnutils.h
#define Knew_createImg(x) KRN_new(x, "createImg","__global unsigned char *out, Vr v, int vx, int vy, int imi, int imy, int k0")
#define Knew_putIMG(x) KRN_new(x, "putIMG","__global unsigned char *imagem_saida, Vr v, int z, REAL px, REAL py, int imy, int width, int i0, int j0, int vx, int vy, int k0")
#define Knew_normalizeVector(x) KRN_new(x, "normalizeVector","Vr input, Vr saida, REAL multiplicador, REAL somador, REAL subtrator, int k0")
#define Knew_kernel_sub(x) KRN_new(x, "kernel_sub","Vr ds, Vr s, Vr t, int k0")
#define Knew_kernel_normalizechar2real(x) KRN_new(x, "kernel_normalizechar2real","Vr dst, __global unsigned char *src, REAL a, REAL b, int k0")
#define Knew_kernel_getVetorClassFromChar(x) KRN_new(x, "kernel_getVetorClassFromChar","Vr dst, __global unsigned char *ints, unsigned int noptiobs, int k0")
#define Knew_kernel_fixW(x) KRN_new(x, "kernel_fixW","Vr w, Vr dw, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0")

//conv.h
#define Knew_convSum(x) KRN_new(x, "convSum","Vr filtro, Vr entrada, Vr saida, int passox, int passoy, int saidatx, int saidaty, int entradatx, int entradaty, int fx, int fy, int fz, int k0")
#define Knew_convCalcGradAndFixWeight(x) KRN_new(x, "convCalcGradAndFixWeight","Vr filtros, Vr ds, Vr entrada, Vr gradFiltro, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, REAL hitLearn, REAL momento, REAL weightDecay, int k0")
#define Knew_convCalcGradIn(x) KRN_new(x, "convCalcGradIn","Vr filtro, Vr gradEntrada, Vr gradNext, int fx, int fy, int fz, int passox, int passoy, int entradatx, int entradaty, int saidatx, int saidaty, int saidatz, int k0")
#define Knew_convCalcGradBatch(x) KRN_new(x, "convCalcGradBatch","Vr ds, Vr entrada, Vr gradFiltro, long batchSize, int fx, int fy, int fz, int entrada_tx, int entrada_ty, int saida_tx, int saida_ty, int passox, int passoy, int k0")

//conv2d.h
#define Knew_conv2dSum(x) KRN_new(x, "conv2dSum","Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int ax, int ay, int az, int fx, int fy, int fz, int fid, int k0")
#define Knew_conv2dCalcGradZ(x) KRN_new(x, "conv2dCalcGradZ","Vr ds, Vr z, Vw dz, int fid, int k0")
#define Knew_conv2dCalcGradIn(x) KRN_new(x, "conv2dCalcGradIn","Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int az, int sx, int sy,  int k0")
#define Knew_conv2dCalcGradAndFixWeight(x) KRN_new(x, "conv2dCalcGradAndFixWeight","Vrw W, Vr dz, Vr a, Vrw dW, int fx, int fy, int ax, int ay, int az, int sx, int sy, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0")
#define Knew_conv2dCalcGradBatch(x) KRN_new(x, "conv2dCalcGradBatch","Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int ax, int ay, int az, int sx, int sy, int px, int py, int k0")

//convf.h
#define Knew_convFSum(x) KRN_new(x, "convFSum","Vr W, Vr a, Vw Z, Vw s, int px, int py, int sx, int sy, int atx, int aty, int fx, int fy, int fz, int fid, int k0")
#define Knew_convFCalcGradZ(x) KRN_new(x, "convFCalcGradZ","Vr ds, Vr z, Vw dz, int fid, int k0")
#define Knew_convFCalcGradIn(x) KRN_new(x, "convFCalcGradIn","Vr W, Vw da, Vr dz, int fx, int fy, int fz, int px, int py, int atx, int aty, int sx, int sy, int sz, int k0")
#define Knew_convFCalcGradAndFixWeight(x) KRN_new(x, "convFCalcGradAndFixWeight","Vr W, Vr dz, Vr a, Vr gradW, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, REAL hitLearn, REAL momento, REAL weightDecay, int k0")
#define Knew_convFCalcGradBatch(x) KRN_new(x, "convFCalcGradBatch","Vr dz, Vr a, Vr dW, long batchSize, int fx, int fy, int fz, int a_tx, int a_ty, int s_tx, int s_ty, int px, int py, int k0")

//convNc.h
#define Knew_convncSum(x) KRN_new(x, "convncSum","Vr W, Vr A, Vr Z, Vr S, unsigned int fid, unsigned int passox, int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0")
#define Knew_convncCalcGradZ(x) KRN_new(x, "convncCalcGradZ","Vr ds, Vr z, Vr dz, unsigned int fid, int k0")
#define Knew_convncCalcGrads(x) KRN_new(x, "convncCalcGrads","Vr W, Vr DA, Vr dz, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, unsigned int entradatx, unsigned int entradaty, unsigned int saidatx, unsigned int saidaty, unsigned int fx, unsigned int fy, unsigned int fz, int k0")
#define Knew_convncCalcFiltro(x) KRN_new(x, "convncCalcFiltro","Vr dz, Vr A, Vr W, Vr dW, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, REAL hitlearn, REAL momento, REAL weightDecay, int k0")
#define Knew_convncCalcFiltroBatch(x) KRN_new(x, "convncCalcFiltroBatch","Vr dz, Vr A, Vr dW, long batchSize, unsigned int dw_x, unsigned int dw_y, unsigned int dw_z, unsigned int a_x, unsigned int a_y, unsigned int s_x, unsigned int s_y, unsigned int passox, unsigned int passoy, unsigned int largx, unsigned int largy, int k0")

//dropout.h
#define Knew_dropativa(x) KRN_new(x, "dropativa","Vr entrada, Vr saida, __global char *hitmap, long seed, REAL pativa, int k0")
#define Knew_dropcalcgrad(x) KRN_new(x, "dropcalcgrad","Vr gradentrada, __global char *hitmap, Vr gradnext, int k0")

//fullconnect.h
#define Knew_fullfeed(x) KRN_new(x, "fullfeed","Vr a, Vr w, Vr b, Vr z, Vr s, int fid, int w_x, int w_y, int k0")
#define Knew_fullCalcDWandFix(x) KRN_new(x, "fullCalcDWandFix","Vr a, Vr w, Vr dw, Vr dz, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int pesosy, int k0")
#define Knew_fullCalcDz(x) KRN_new(x, "fullCalcDz","Vr dz, Vr ds, Vr z, int dfa, int k0")
#define Knew_fullCalcDzBath(x) KRN_new(x, "fullCalcDzBath","Vr dz, Vr ds, Vr z, Vr db, int dfa, long batchSize, int k0")
#define Knew_fullCalcDzAndFixB(x) KRN_new(x, "fullCalcDzAndFixB","Vr dz, Vr ds, Vr z, Vr b, Vr db, int dfa, REAL hitlearn, REAL momento, REAL decaimentoDePeso, int k0")
#define Knew_fullcalcin(x) KRN_new(x, "fullcalcin","Vr dz, Vr da, Vr w, int pesosx, int pesosy, int k0")
#define Knew_fullCalcDWBatch(x) KRN_new(x, "fullCalcDWBatch","Vr a, Vr dw, Vr dz, long batchSize, int pesosy, int k0")

//padding.h
#define Knew_paddingfeed(x) KRN_new(x, "paddingfeed","Vr in, Vr out, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0")
#define Knew_paddingBack(x) KRN_new(x, "paddingBack","Vr gradNext, Vr gradin, unsigned int txi, unsigned int tyi, unsigned int txo, unsigned int tyo, unsigned int t, unsigned int l, int k0")

//poolav.h
#define Knew_poolAVativa(x) KRN_new(x, "poolAVativa","Vr entrada, Vr saida, int passox, int passoy, int fx, int fy, int saidatx, int saidaty, int entradatx, int entradaty, int k0")
#define Knew_poolAvCalcGrads(x) KRN_new(x, "poolAvCalcGrads","Vr entrada, Vr gradEntrada, Vr gradNext, Vr saida, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0")

//poolMax.h
#define Knew_poolativa(x) KRN_new(x, "poolativa","Vr entrada, Vr saida, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0")
#define Knew_poolCalcGrads(x) KRN_new(x, "poolCalcGrads","Vr entrada, Vr gradEntrada, Vr gradNext, Vr saida, int fx, int fy, int px, int py, int entradatx, int entradaty, int saidatx, int saidaty, int k0")

//poolMin.h
#define Knew_poolativaMin(x) KRN_new(x, "poolativaMin","Vr entrada, Vr saida, int passox, int passoy, int filtrox, int filtroy, int saidatx, int saidaty, int entradatx, int entradaty, int k0")

//prelu.h
#define Knew_preluativa(x) KRN_new(x, "preluativa","Vr entrada, Vr saida, Vr A, int k0")
#define Knew_prelucalcgrad(x) KRN_new(x, "prelucalcgrad","Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, int learn, REAL hitlearn, REAL momento, REAL decaimento, int k0")
#define Knew_preluonlyfix(x) KRN_new(x, "preluonlyfix","Vr entrada, Vr gradnext, Vr A, Vr dA, REAL hitlearn, REAL momento, REAL decaimento, int k0")
#define Knew_prelucalcgradBatch(x) KRN_new(x, "prelucalcgradBatch","Vr gradentrada, Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0")
#define Knew_preluonlyDABatch(x) KRN_new(x, "preluonlyDABatch","Vr entrada, Vr gradnext, Vr A, Vr dA, long batchSize, int k0")

//relu.h
#define Knew_reluativa(x) KRN_new(x, "reluativa","Vr entrada, Vr saida, REAL menor, REAL maior, int k0")
#define Knew_relucalcgrad(x) KRN_new(x, "relucalcgrad","Vr gradentrada, Vr entrada, Vr gradnext, REAL menor, REAL maior, int k0")

//softmax.h
#define Knew_softmaxExp(x) KRN_new(x, "softmaxExp","Vr entrada, Vr exponent, int k0")
#define Knew_softmaxSomaExp(x) KRN_new(x, "softmaxSomaExp","Vr eps, Vr soma, int saidatx, int saidaty, int k0")
#define Knew_softmaxNormaliza(x) KRN_new(x, "softmaxNormaliza","Vr exponet, Vr soma, Vr saida, int saidatx, int saidaty, int k0")
#define Knew_softMaxcalcgrad(x) KRN_new(x, "softMaxcalcgrad","Vr da, Vr s, Vr ds, int sx, int sy, int k0")
#define Knew_softmaxFindMax(x) KRN_new(x, "softmaxFindMax","Vr a, Vr mx, __global int *i_max, int ax, int ay, int k0")
#define Knew_softmaxExpNorm(x) KRN_new(x, "softmaxExpNorm","Vr entrada, Vr exponent, Vr mx, int ax, int ay, int k0")

