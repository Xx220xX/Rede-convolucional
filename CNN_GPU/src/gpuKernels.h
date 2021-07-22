#ifndef GAB_KERNELS_OPENCL_H
#define GAB_KERNELS_OPENCL_H
//utils.h
// Created by Xx220xX on 10/05/2020.

#define Vector __global double *

#define kV __kernel void

#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))

#define TensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))

#define TensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\
_y_ = total%ty      ;                                        \
_x_ = (total - _y_)%(ty*tx)/ty ;                             \
_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \
_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);


#define TensorRemap(total, _x_, _y_, _z_, tx, ty)\
_y_ = total % ty;\
_x_ = ((total - _y_) % (ty * tx)) / ty;\
_z_ = (k - _x_ * ty - _y_) / (tx * ty);

#define TensorRemap2D(total, x, y, ty)\
y = total % ty;\
x = total/ ty;

typedef struct {
	int x, y, z;
} Ponto3d;

typedef struct {
	Ponto3d min, max;
} Range;

kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) ;


kV normalizeVector(Vector input, Vector saida, double multiplicador,
				   double somador, double subtrator, int k0) ;


kV subKernel(Vector grad, Vector saida, Vector target, int k0) ;

kV divKernel(Vector v, double value, int k0) ;

kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) ;

kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) ;



//bathnorm.h

// achar a media
kV BatchNormMedia(Vector entrada, Vector media,
				  int entradatx, int entradaty, int k0) ;

// achar a diferenca
kV BatchNormDiferenca(Vector entrada, Vector media,
					  Vector diferenca,
					  Vector diferencaquad,
					  int entradatx, int entradaty, int k0) ;

kV BatchNormVariance(Vector dif, Vector difQuad,
					 Vector sumdiferenca, Vector variancia,
					 double episolon, int diftx, int difty,
					 int k0) ;

// normaliza
kV BatchNormNormaliza(Vector saida,
					  Vector norma,
					  Vector diferenca,
					  Vector variancia,
					  Vector Y,
					  Vector B,
					  int diferencatx, int diferencaty, int k0) ;


kV BatchNormaCalcGrad1(Vector gradIn,
					   Vector gradNext,
					   Vector variancia,
					   Vector media,
					   Vector Y,

					   Vector somaDif,
					   Vector entrada,
					   int entradatx,
					   int entradaty,
					   int k0) ;

kV BatchNormaCalcGrad2(Vector gradNext,
					   Vector norma,
					   Vector gradY,
					   Vector gradB,
					   int entradatx,
					   int entradaty,
					   int k0) ;


kV batchNormCorrigePeso(Vector gradY,
						Vector gradB,
						Vector Y,
						Vector B,
						double hitlearn,
						int k0) ;
//conv.h
kV convSum(Vector filtro, Vector entrada, Vector saida,
		   int passox, int passoy,
		   int saidatx, int saidaty,
		   int entradatx, int entradaty,
		   int fx, int fy, int fz, int k0) ;


kV convCalcGradAndFixWeight(Vector filtros, Vector ds,
							Vector entrada, Vector gradFiltro,
							int fx, int fy, int fz,
							int entrada_tx, int entrada_ty,
							int saida_tx, int saida_ty,
							int passox, int passoy,
							double hitLearn, double momento, double weightDecay,
							int k0) ;

kV convCalcGradIn(Vector filtro, Vector gradEntrada, Vector gradNext,
				  int fx, int fy, int fz,
				  int passox, int passoy,
				  int entradatx, int entradaty,
				  int saidatx, int saidaty, int saidatz,
				  int k0) ;


//convNc.h
//#include"utils.h"
kV convncSum(Vector filtro, Vector entrada, Vector saida,
			 int passox, int passoy, int largx,
			 int largy, int saidatx, int saidaty,
			 int entradatx, int entradaty, int fx, int fy,
			 int entradatz, int k0) ;

kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,
				   double hitlearn,
				   double momento, double weightDecay, int k0) ;

kV convncCalcFiltro(Vector ds,
					Vector entrada,
					Vector gradFiltro,
					int gradFiltro_tx,
					int gradFiltro_ty,
					int gradFiltro_tz,

					int entrada_tx,
					int entrada_ty,

					int saida_tx,
					int saida_ty,

					int passox,
					int passoy,

					int largx,
					int largy,
					int k0) ;


kV convncCalcGrads(Vector filtro,
				   Vector entrada,
				   Vector gradEntrada,
				   Vector gradNext,

				   int passox,
				   int passoy,
				   int largx,
				   int largy,

				   int entradatx,
				   int entradaty,
				   int saidatx,
				   int saidaty,

				   int fx,
				   int fy,
				   int fz,
				   int numFilters,

				   int k0) ;


//dropout.h
#define MAX_INT_DP  ((1UL << 31) - 1)

long randoml(unsigned long seed, unsigned long id) ;

double randomD(unsigned long seed, unsigned long id) ;

kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,
			 double pativa, int k0) ;


kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) ;

//fullconnect.h
double sigmoid(double x) ;

double difsigmoid(double x) ;

double tanghG(double x) ;

double diftanhG(double x) ;

double relu(double x) ;

double difrelu(double x) ;

double func(int id, double x) ;

kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,
			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) ;

kV fullfixweight(Vector a,
				 Vector pesos,
				 Vector dw,
				 Vector dz,
				 double hitlearn,
				 double decaimentoDePeso,
				 double momento,
				 int pesosy,
				 int k0) ;

kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) ;

kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,
				  int k0) ;

//padding.h
kV paddingfeed(Vector in, Vector out,
			   int txi, int tyi,
			   int txo, int tyo,
			   int t, int l,
			   int k0) ;

kV paddingBack(Vector gradNext, Vector gradin,
			   int txi, int tyi,
			   int txo, int tyo,
			   int t, int l, int k0) ;
//pool.h
kV poolativa(Vector entrada, Vector saida, int lenFilter,
			 int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) ;


kV poolCalcGrads(Vector entrada, Vector gradEntrada,
				 Vector gradNext, Vector saida,
				 int fx, int fy, int px, int py,
				 int entradatx, int entradaty,
				 int saidatx, int saidaty,
				 int k0) ;


//poolav.h
kV PoolAvativa(Vector entrada, Vector saida, int lenFilter,
			   int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) ;


kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,
				   int fx, int fy, int px, int py,
				   int entradatx, int entradaty, int saidatx, int saidaty,
				   int k0) ;


//relu.h
kV reluativa(Vector entrada, Vector saida, int k0) ;

kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) ;

//softmax.h
kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,
				 int entradaty,
				 int k0) ;

kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,
				 int saidatx, int saidaty, int k0) ;

kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) ;


#endif //GAB_KERNELS_OPENCL_H
