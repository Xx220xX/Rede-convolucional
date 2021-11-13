//
// Created by hslhe on 14/08/2021.
//
#include "utils/defaultkernel.h"
const char __default_kernel__[] = 
/*1*/		"#ifndef GAB_KERNELS_OPENCL_H\n"
/*2*/		"#define GAB_KERNELS_OPENCL_H\n"
/*3*/		"//utils.h\n"
/*4*/		"// Created by Xx220xX on 10/05/2020.\n"
/*5*/		"#ifndef ATIVATIONSFUNCTIONS_H\n"
/*6*/		"#define ATIVATIONSFUNCTIONS_H\n"
/*7*/		"#define  REAL float\n"
/*8*/		"\n"
/*9*/		"#define Vector __global REAL *\n"
/*10*/		"\n"
/*11*/		"#define kV __kernel void\n"
/*12*/		"\n"
/*13*/		"#define KTensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))\n"
/*14*/		"\n"
/*15*/		"#define KTensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))\n"
/*16*/		"\n"
/*17*/		"#define KTensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\\\n"
/*18*/		"_y_ = total%ty      ;                                        \\\n"
/*19*/		"_x_ = (total - _y_)%(ty*tx)/ty ;                             \\\n"
/*20*/		"_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \\\n"
/*21*/		"_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);\n"
/*22*/		"\n"
/*23*/		"\n"
/*24*/		"#define KTensorRemap(total, _x_, _y_, _z_, tx, ty)\\\n"
/*25*/		"_y_ = total % ty;\\\n"
/*26*/		"_x_ = ((total - _y_) % (ty * tx)) / ty;\\\n"
/*27*/		"_z_ = (k - _x_ * ty - _y_) / (tx * ty);\n"
/*28*/		"\n"
/*29*/		"#define KTensorRemap2D(total, x, y, ty)\\\n"
/*30*/		"y = total % ty;\\\n"
/*31*/		"x = total/ ty;\n"
/*32*/		"\n"
/*33*/		"typedef struct {\n"
/*34*/		"	int x, y, z;\n"
/*35*/		"} Ponto3d;\n"
/*36*/		"\n"
/*37*/		"typedef struct {\n"
/*38*/		"	Ponto3d min, max;\n"
/*39*/		"} Range;\n"
/*40*/		"\n"
/*41*/		"\n"
/*42*/		"\n"
/*43*/		"REAL sigmoid(REAL x) {\n"
/*44*/		"	return 1.0 / (1.0 + exp(-x));\n"
/*45*/		"}\n"
/*46*/		"\n"
/*47*/		"REAL difsigmoid(REAL x) {\n"
/*48*/		"	REAL tmp = sigmoid(x);\n"
/*49*/		"	return tmp * (1.0 - tmp);\n"
/*50*/		"}\n"
/*51*/		"\n"
/*52*/		"REAL tanghG(REAL x) {\n"
/*53*/		"	return tanh(x);\n"
/*54*/		"}\n"
/*55*/		"\n"
/*56*/		"REAL diftanhG(REAL x) {\n"
/*57*/		"	REAL tmp = tanh(x);\n"
/*58*/		"	return (1.0 - tmp * tmp);\n"
/*59*/		"}\n"
/*60*/		"\n"
/*61*/		"REAL relu(REAL x) {\n"
/*62*/		"	return x > 0 ? x : 0.0;\n"
/*63*/		"}\n"
/*64*/		"\n"
/*65*/		"REAL difrelu(REAL x) {\n"
/*66*/		"	return x > 0 ? 1.0 : 0.0;\n"
/*67*/		"}\n"
/*68*/		"\n"
/*69*/		"REAL func(int id, REAL x) {\n"
/*70*/		"	switch (id) {\n"
/*71*/		"		case 0:\n"
/*72*/		"			return sigmoid(x);\n"
/*73*/		"		case 1:\n"
/*74*/		"			return difsigmoid(x);\n"
/*75*/		"		case 2:\n"
/*76*/		"			return tanghG(x);\n"
/*77*/		"		case 3:\n"
/*78*/		"			return diftanhG(x);\n"
/*79*/		"		case 4:\n"
/*80*/		"			return relu(x);\n"
/*81*/		"		case 5:\n"
/*82*/		"			return difrelu(x);\n"
/*83*/		"		case 6:\n"
/*84*/		"			return x;\n"
/*85*/		"		case 7:\n"
/*86*/		"			return 1;\n"
/*87*/		"		default:\n"
/*88*/		"			return 0;\n"
/*89*/		"	}\n"
/*90*/		"}\n"
/*91*/		"\n"
/*92*/		"#endif\n"
/*93*/		"//bathnorm.h\n"
/*94*/		"\n"
/*95*/		"// achar a media\n"
/*96*/		"kV BatchNormMedia(Vector entrada, Vector media,\n"
/*97*/		"                  int entradatx, int entradaty, int k0) {\n"
/*98*/		"	int z = get_global_id(0) + k0;\n"
/*99*/		"	int x, y;\n"
/*100*/		"	REAL m = 0;\n"
/*101*/		"	for (x = 0; x < entradatx; x++) {\n"
/*102*/		"		for (y = 0; y < entradaty; y++) {\n"
/*103*/		"			m += entrada[KTensorMap(x, y, z, entradatx, entradaty)];\n"
/*104*/		"		}\n"
/*105*/		"	}\n"
/*106*/		"	media[z] = m / (REAL) (entradatx * entradaty);\n"
/*107*/		"}\n"
/*108*/		"\n"
/*109*/		"// achar a diferenca\n"
/*110*/		"kV BatchNormDiferenca(Vector entrada, Vector media,\n"
/*111*/		"                      Vector diferenca,\n"
/*112*/		"                      Vector diferencaquad,\n"
/*113*/		"                      int entradatx, int entradaty, int k0) {\n"
/*114*/		"	int x, y, z;\n"
/*115*/		"	int k = get_global_id(0) + k0;\n"
/*116*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*117*/		"	diferenca[k] = entrada[k] - media[z];\n"
/*118*/		"	diferencaquad[k] = diferenca[k] * diferenca[k];\n"
/*119*/		"}\n"
/*120*/		"\n"
/*121*/		"kV BatchNormVariance(Vector dif, Vector difQuad,\n"
/*122*/		"                     Vector sumdiferenca, Vector variancia,\n"
/*123*/		"                     REAL episolon, int diftx, int difty,\n"
/*124*/		"                     int k0) {\n"
/*125*/		"	int z = get_global_id(0) + k0;\n"
/*126*/		"	REAL sum = 0;\n"
/*127*/		"	REAL sumdif = 0;\n"
/*128*/		"	for (int x = 0; x < diftx; x++) {\n"
/*129*/		"		for (int y = 0; y < difty; y++) {\n"
/*130*/		"			sum += difQuad[KTensorMap(x, y, z, diftx, difty)];\n"
/*131*/		"			sumdif += dif[KTensorMap(x, y, z, diftx, difty)];\n"
/*132*/		"		}\n"
/*133*/		"	}\n"
/*134*/		"	sumdiferenca[z] = sumdif;\n"
/*135*/		"	variancia[z] = sqrt(sum / (difty * diftx) + episolon);\n"
/*136*/		"}\n"
/*137*/		"\n"
/*138*/		"// normaliza\n"
/*139*/		"kV BatchNormNormaliza(Vector saida,\n"
/*140*/		"                      Vector norma,\n"
/*141*/		"                      Vector diferenca,\n"
/*142*/		"                      Vector variancia,\n"
/*143*/		"                      Vector Y,\n"
/*144*/		"                      Vector B,\n"
/*145*/		"                      int diferencatx, int diferencaty, int k0) {\n"
/*146*/		"	int x, y, z;\n"
/*147*/		"	int k = get_global_id(0) + k0;\n"
/*148*/		"	KTensorRemap(k, x, y, z, diferencatx, diferencaty)\n"
/*149*/		"	norma[k] = diferenca[k] / variancia[z];\n"
/*150*/		"	saida[k] = norma[k] * Y[z] + B[z];\n"
/*151*/		"}\n"
/*152*/		"\n"
/*153*/		"\n"
/*154*/		"kV BatchNormaCalcGrad1(Vector gradIn,\n"
/*155*/		"                       Vector gradNext,\n"
/*156*/		"                       Vector variancia,\n"
/*157*/		"                       Vector media,\n"
/*158*/		"                       Vector Y,\n"
/*159*/		"\n"
/*160*/		"                       Vector somaDif,\n"
/*161*/		"                       Vector entrada,\n"
/*162*/		"                       int entradatx,\n"
/*163*/		"                       int entradaty,\n"
/*164*/		"                       int k0) {\n"
/*165*/		"	int x, y, z;\n"
/*166*/		"	int k = get_global_id(0) + k0;\n"
/*167*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*168*/		"	REAL M = entradatx * entradaty;\n"
/*169*/		"	REAL dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);\n"
/*170*/		"	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);\n"
/*171*/		"\n"
/*172*/		"	REAL didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;\n"
/*173*/		"	didx = didx / (variancia[z] * variancia[z]);\n"
/*174*/		"	didx = didx * gradNext[k];\n"
/*175*/		"	gradIn[k] = didx * Y[z];\n"
/*176*/		"}\n"
/*177*/		"\n"
/*178*/		"kV BatchNormaCalcGrad2(Vector gradNext,\n"
/*179*/		"                       Vector norma,\n"
/*180*/		"                       Vector gradY,\n"
/*181*/		"                       Vector gradB,\n"
/*182*/		"                       int entradatx,\n"
/*183*/		"                       int entradaty,\n"
/*184*/		"                       int k0) {\n"
/*185*/		"	int z = get_global_id(0) + k0;\n"
/*186*/		"	REAL sumY = 0;\n"
/*187*/		"	REAL sumB = 0;\n"
/*188*/		"	int k;\n"
/*189*/		"	for (int x = 0; x < entradatx; ++x) {\n"
/*190*/		"		for (int y = 0; y < entradaty; ++y) {\n"
/*191*/		"			k = KTensorMap(x, y, z, entradatx, entradaty);\n"
/*192*/		"			sumY += gradNext[k];\n"
/*193*/		"			sumB += gradNext[k] * norma[k];\n"
/*194*/		"		}\n"
/*195*/		"	}\n"
/*196*/		"	gradB[z] = sumB;\n"
/*197*/		"	gradY[z] = sumY;\n"
/*198*/		"}\n"
/*199*/		"\n"
/*200*/		"\n"
/*201*/		"kV batchNormCorrigePeso(Vector gradY,\n"
/*202*/		"                        Vector gradB,\n"
/*203*/		"                        Vector Y,\n"
/*204*/		"                        Vector B,\n"
/*205*/		"                        REAL hitlearn,\n"
/*206*/		"                        int k0) {\n"
/*207*/		"	int z = get_global_id(0) + k0;\n"
/*208*/		"	B[z] = B[z] - gradB[z] * hitlearn;\n"
/*209*/		"	Y[z] = Y[z] - gradY[z] * hitlearn;\n"
/*210*/		"}\n"
/*211*/		"//cnnutils.h\n"
/*212*/		"//\n"
/*213*/		"// Created by Henrique on 22-Jul-21.\n"
/*214*/		"//\n"
/*215*/		"\n"
/*216*/		"\n"
/*217*/		"kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {\n"
/*218*/		"	int k = get_global_id(0) + k0;\n"
/*219*/		"	int i, j, z;\n"
/*220*/		"	KTensorRemap(k, i, j, z, vx, vy)\n"
/*221*/		"	imi = imi + i;\n"
/*222*/		"	int imj = j + z * vy + z;\n"
/*223*/		"	out[imi * imy + imj] = ((int) v[k]) & 0xff;\n"
/*224*/		"}\n"
/*225*/		"\n"
/*226*/		"kV putIMG(__global unsigned char *imagem_saida,\n"
/*227*/		"		  Vector v,\n"
/*228*/		"		  int z,\n"
/*229*/		"		  REAL px,\n"
/*230*/		"		  REAL py,\n"
/*231*/		"		  int imy,\n"
/*232*/		"		  int width,\n"
/*233*/		"		  int i0,\n"
/*234*/		"		  int j0,\n"
/*235*/		"		  int vx,\n"
/*236*/		"		  int vy,\n"
/*237*/		"		  int k0) {\n"
/*238*/		"	int k = get_global_id(0) + k0;\n"
/*239*/		"	int i, j;\n"
/*240*/		"	KTensorRemap2D(k, i, j, imy)\n"
/*241*/		"	int x = i * px, y = j * py;\n"
/*242*/		"	imagem_saida[(i+i0)*width+j+j0] = ((int) v[KTensorMap(x, y, z, vx, vy)] ) & 0xff;\n"
/*243*/		"}\n"
/*244*/		"\n"
/*245*/		"\n"
/*246*/		"kV normalizeVector(Vector input, Vector saida, REAL multiplicador, REAL somador, REAL subtrator,\n"
/*247*/		"				   int k0) {\n"
/*248*/		"	int k = get_global_id(0) + k0;\n"
/*249*/		"	saida[k] = (input[k] + somador) * multiplicador - subtrator;\n"
/*250*/		"}\n"
/*251*/		"\n"
/*252*/		"\n"
/*253*/		"kV subKernel(Vector grad, Vector saida, Vector target, int k0) {\n"
/*254*/		"	int k = get_global_id(0) + k0;\n"
/*255*/		"	grad[k] = saida[k] - target[k];\n"
/*256*/		"}\n"
/*257*/		"\n"
/*258*/		"kV divKernel(Vector v, REAL value, int k0) {\n"
/*259*/		"	int k = get_global_id(0) + k0;\n"
/*260*/		"	v[k] = v[k] / value;\n"
/*261*/		"}\n"
/*262*/		"\n"
/*263*/		"kV divIntDo(__global unsigned char *src, Vector v, REAL value, int k0) {\n"
/*264*/		"	int k = get_global_id(0) + k0;\n"
/*265*/		"	v[k] = ((REAL) src[k]) / value;\n"
/*266*/		"\n"
/*267*/		"}\n"
/*268*/		"\n"
/*269*/		"kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {\n"
/*270*/		"	int w = get_global_id(0) + k0;\n"
/*271*/		"	int y = ints[w];\n"
/*272*/		"	v[KTensorMap4D(0, y, 0, w, 1, noptiobs, 1)] = 1.0;\n"
/*273*/		"}\n"
/*274*/		"\n"
/*275*/		"//conv.h\n"
/*276*/		"kV convSum(Vector filtro, Vector entrada, Vector saida,\n"
/*277*/		"           int passox, int passoy,\n"
/*278*/		"           int saidatx, int saidaty,\n"
/*279*/		"           int entradatx, int entradaty,\n"
/*280*/		"           int fx, int fy, int fz, int k0) {\n"
/*281*/		"	int k = get_global_id(0) + k0;\n"
/*282*/		"	int x, y, filtrok;\n"
/*283*/		"	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*284*/		"	REAL sum = 0, f = 0, v = 0;\n"
/*285*/		"	int lf = 0, le = 0;\n"
/*286*/		"	for (int m = 0; m < fx; m++) {\n"
/*287*/		"		for (int n = 0; n < fy; n++) {\n"
/*288*/		"			for (int z = 0; z < fz; z++) {\n"
/*289*/		"				lf = KTensorMap4D(m, n, z, filtrok, fx, fy, fz);\n"
/*290*/		"				le = KTensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);\n"
/*291*/		"				f = filtro[lf];\n"
/*292*/		"				v = entrada[le];\n"
/*293*/		"				sum += f * v;\n"
/*294*/		"			}\n"
/*295*/		"		}\n"
/*296*/		"	}\n"
/*297*/		"	saida[k] = sum;\n"
/*298*/		"}\n"
/*299*/		"\n"
/*300*/		"\n"
/*301*/		"kV convCalcGradAndFixWeight(Vector filtros, Vector ds,\n"
/*302*/		"                            Vector entrada, Vector gradFiltro,\n"
/*303*/		"                            int fx, int fy, int fz,\n"
/*304*/		"                            int entrada_tx, int entrada_ty,\n"
/*305*/		"                            int saida_tx, int saida_ty,\n"
/*306*/		"                            int passox, int passoy,\n"
/*307*/		"                            REAL hitLearn, REAL momento, REAL weightDecay,\n"
/*308*/		"                            int k0) {\n"
/*309*/		"	int k = get_global_id(0) + k0;\n"
/*310*/		"	int m, n, z, l;\n"
/*311*/		"	KTensorRemap4D(k, m, n, z, l, fx, fy, fz)\n"
/*312*/		"	REAL soma = 0;\n"
/*313*/		"	int le, ls;\n"
/*314*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*315*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*316*/		"			le = KTensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);\n"
/*317*/		"			ls = KTensorMap(i, j, l, saida_tx, saida_ty);\n"
/*318*/		"			soma += entrada[le]\n"
/*319*/		"			        * ds[ls];\n"
/*320*/		"		}\n"
/*321*/		"	}\n"
/*322*/		"	REAL dw = soma + gradFiltro[k] * momento;\n"
/*323*/		"	REAL w = filtros[k];\n"
/*324*/		"	filtros[k] = w - hitLearn * (dw + w * weightDecay);\n"
/*325*/		"	gradFiltro[k] = dw;\n"
/*326*/		"}\n"
/*327*/		"\n"
/*328*/		"kV convCalcGradIn(Vector filtro, Vector gradEntrada, Vector gradNext,\n"
/*329*/		"                  int fx, int fy, int fz,\n"
/*330*/		"                  int passox, int passoy,\n"
/*331*/		"                  int entradatx, int entradaty,\n"
/*332*/		"                  int saidatx, int saidaty, int saidatz,\n"
/*333*/		"                  int k0) {\n"
/*334*/		"	int k = get_global_id(0) + k0;\n"
/*335*/		"	int x, y, z;\n"
/*336*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*337*/		"\n"
/*338*/		"	Range range_filtro;\n"
/*339*/		"	range_filtro.min.x = 0;\n"
/*340*/		"	if (x + fx > entradatx) {\n"
/*341*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*342*/		"	}\n"
/*343*/		"	range_filtro.max.x = fx - 1;\n"
/*344*/		"	if (x - fx + 1 < 0) {\n"
/*345*/		"		range_filtro.max.x = x;\n"
/*346*/		"	}\n"
/*347*/		"	range_filtro.min.y = 0;\n"
/*348*/		"	if (y + fy > entradaty) {\n"
/*349*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*350*/		"	}\n"
/*351*/		"	range_filtro.max.y = fy - 1;\n"
/*352*/		"	if (y - fy + 1 < 0) {\n"
/*353*/		"		range_filtro.max.y = y;\n"
/*354*/		"	}\n"
/*355*/		"	REAL somaErro = 0, pesoAplicado = 0;\n"
/*356*/		"	int i, j;\n"
/*357*/		"	int lf, ls;\n"
/*358*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*359*/		"		i = (x - m) / passox;\n"
/*360*/		"		if (i * passox + m != x) continue;\n"
/*361*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*362*/		"			j = (y - n) / passoy;\n"
/*363*/		"			if (j * passoy + n != y) continue;\n"
/*364*/		"			for (int w = 0; w < saidatz; w++) {\n"
/*365*/		"				lf = KTensorMap4D(m, n, z, w, fx, fy, fz);\n"
/*366*/		"				ls = KTensorMap(i, j, w, saidatx, saidaty);\n"
/*367*/		"				pesoAplicado = filtro[lf];\n"
/*368*/		"				somaErro += pesoAplicado * gradNext[ls];\n"
/*369*/		"			}\n"
/*370*/		"		}\n"
/*371*/		"	}\n"
/*372*/		"	gradEntrada[k] = somaErro;\n"
/*373*/		"}\n"
/*374*/		"\n"
/*375*/		"\n"
/*376*/		"//convf.h\n"
/*377*/		"kV convFSum(Vector filtro, Vector entrada, Vector Z, Vector saida,\n"
/*378*/		"           int passox, int passoy,\n"
/*379*/		"           int saidatx, int saidaty,\n"
/*380*/		"           int entradatx, int entradaty,\n"
/*381*/		"           int fx, int fy, int fz, int fid, int k0) {\n"
/*382*/		"	int k = get_global_id(0) + k0;\n"
/*383*/		"	int x, y, filtrok;\n"
/*384*/		"\n"
/*385*/		"	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*386*/		"	REAL sum = 0, f , v ;\n"
/*387*/		"	int lf, le ;\n"
/*388*/		"	for (int m = 0; m < fx; m++) {\n"
/*389*/		"		for (int n = 0; n < fy; n++) {\n"
/*390*/		"			for (int z = 0; z < fz; z++) {\n"
/*391*/		"				lf = KTensorMap4D(m, n, z, filtrok, fx, fy, fz);\n"
/*392*/		"				le = KTensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);\n"
/*393*/		"				f = filtro[lf];\n"
/*394*/		"				v = entrada[le];\n"
/*395*/		"				sum += f * v;\n"
/*396*/		"			}\n"
/*397*/		"		}\n"
/*398*/		"	}\n"
/*399*/		"	Z[k] = sum;\n"
/*400*/		"	saida[k] = func(fid,sum);\n"
/*401*/		"}\n"
/*402*/		"\n"
/*403*/		"kV convFCalcGradZ(Vector  ds,Vector z,Vector dz,int fid,int k0){\n"
/*404*/		"	int k = get_global_id(0) + k0;\n"
/*405*/		"	dz[k] = ds[k]*func(fid,z[k]);\n"
/*406*/		"}\n"
/*407*/		"kV convFCalcGradAndFixWeight(Vector filtros, Vector dz,\n"
/*408*/		"                            Vector entrada, Vector gradFiltro,\n"
/*409*/		"                            int fx, int fy, int fz,\n"
/*410*/		"                            int entrada_tx, int entrada_ty,\n"
/*411*/		"                            int saida_tx, int saida_ty,\n"
/*412*/		"                            int passox, int passoy,\n"
/*413*/		"                            REAL hitLearn, REAL momento, REAL weightDecay,\n"
/*414*/		"                            int k0) {\n"
/*415*/		"	int k = get_global_id(0) + k0;\n"
/*416*/		"	int m, n, z, l;\n"
/*417*/		"	KTensorRemap4D(k, m, n, z, l, fx, fy, fz)\n"
/*418*/		"	REAL soma = 0;\n"
/*419*/		"	int le, ls;\n"
/*420*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*421*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*422*/		"			le = KTensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);\n"
/*423*/		"			ls = KTensorMap(i, j, l, saida_tx, saida_ty);\n"
/*424*/		"			soma += entrada[le]\n"
/*425*/		"			        * dz[ls];\n"
/*426*/		"		}\n"
/*427*/		"	}\n"
/*428*/		"	REAL dw = soma + gradFiltro[k] * momento;\n"
/*429*/		"	REAL w = filtros[k];\n"
/*430*/		"	filtros[k] = w - hitLearn * (dw + w * weightDecay);\n"
/*431*/		"	gradFiltro[k] = dw;\n"
/*432*/		"}\n"
/*433*/		"\n"
/*434*/		"kV convFCalcGradIn(Vector filtro, Vector gradEntrada, Vector dz,\n"
/*435*/		"                  int fx, int fy, int fz,\n"
/*436*/		"                  int passox, int passoy,\n"
/*437*/		"                  int entradatx, int entradaty,\n"
/*438*/		"                  int saidatx, int saidaty, int saidatz,\n"
/*439*/		"                  int k0) {\n"
/*440*/		"	int k = get_global_id(0) + k0;\n"
/*441*/		"	int x, y, z;\n"
/*442*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*443*/		"\n"
/*444*/		"	Range range_filtro;\n"
/*445*/		"	range_filtro.min.x = 0;\n"
/*446*/		"	if (x + fx > entradatx) {\n"
/*447*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*448*/		"	}\n"
/*449*/		"	range_filtro.max.x = fx - 1;\n"
/*450*/		"	if (x - fx + 1 < 0) {\n"
/*451*/		"		range_filtro.max.x = x;\n"
/*452*/		"	}\n"
/*453*/		"	range_filtro.min.y = 0;\n"
/*454*/		"	if (y + fy > entradaty) {\n"
/*455*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*456*/		"	}\n"
/*457*/		"	range_filtro.max.y = fy - 1;\n"
/*458*/		"	if (y - fy + 1 < 0) {\n"
/*459*/		"		range_filtro.max.y = y;\n"
/*460*/		"	}\n"
/*461*/		"	REAL somaErro = 0, pesoAplicado = 0;\n"
/*462*/		"	int i, j;\n"
/*463*/		"	int lf, ls;\n"
/*464*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*465*/		"		i = (x - m) / passox;\n"
/*466*/		"		if (i * passox + m != x) continue;\n"
/*467*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*468*/		"			j = (y - n) / passoy;\n"
/*469*/		"			if (j * passoy + n != y) continue;\n"
/*470*/		"			for (int w = 0; w < saidatz; w++) {\n"
/*471*/		"				lf = KTensorMap4D(m, n, z, w, fx, fy, fz);\n"
/*472*/		"				ls = KTensorMap(i, j, w, saidatx, saidaty);\n"
/*473*/		"				pesoAplicado = filtro[lf];\n"
/*474*/		"				somaErro += pesoAplicado * dz[ls];\n"
/*475*/		"			}\n"
/*476*/		"		}\n"
/*477*/		"	}\n"
/*478*/		"	gradEntrada[k] = somaErro;\n"
/*479*/		"}\n"
/*480*/		"\n"
/*481*/		"\n"
/*482*/		"//convNc.h\n"
/*483*/		"//#include\"utils.h\"\n"
/*484*/		"kV convncSum(Vector filtro, Vector entrada, Vector saida,\n"
/*485*/		"             int passox, int passoy, int largx,\n"
/*486*/		"             int largy, int saidatx, int saidaty,\n"
/*487*/		"             int entradatx, int entradaty,int fx, int fy,\n"
/*488*/		"             int entradatz, int k0) {\n"
/*489*/		"	int k = get_global_id(0) + k0;\n"
/*490*/		"	int x, y, filtrok;\n"
/*491*/		"	KTensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*492*/		"	Ponto3d Kmapeado = {x * passox, y * passoy, 0};\n"
/*493*/		"	REAL sum = 0, f, v;\n"
/*494*/		"	for (int i = 0; i < fx; i++)\n"
/*495*/		"		for (int j = 0; j < fy; j++)\n"
/*496*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*497*/		"				f = filtro[KTensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];\n"
/*498*/		"				v = entrada[KTensorMap(Kmapeado.x + i * largx, Kmapeado.y + j * largy, z, entradatx, entradaty)];\n"
/*499*/		"\n"
/*500*/		"				sum += f * v;\n"
/*501*/		"			}\n"
/*502*/		"	saida[k] = sum;\n"
/*503*/		"}\n"
/*504*/		"\n"
/*505*/		"kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,\n"
/*506*/		"				   REAL hitlearn,\n"
/*507*/		"                   REAL momento, REAL weightDecay, int k0) {\n"
/*508*/		"	int k = get_global_id(0) + k0;\n"
/*509*/		"	REAL m = grad[k] + gradOld[k] * momento;\n"
/*510*/		"	REAL w = filtro[k];\n"
/*511*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*512*/		"	gradOld[k] = m;\n"
/*513*/		"}\n"
/*514*/		"\n"
/*515*/		"kV convncCalcFiltro(Vector ds,\n"
/*516*/		"                    Vector entrada,\n"
/*517*/		"                    Vector gradFiltro,\n"
/*518*/		"                    int gradFiltro_tx,\n"
/*519*/		"                    int gradFiltro_ty,\n"
/*520*/		"                    int gradFiltro_tz,\n"
/*521*/		"\n"
/*522*/		"                    int entrada_tx,\n"
/*523*/		"                    int entrada_ty,\n"
/*524*/		"\n"
/*525*/		"                    int saida_tx,\n"
/*526*/		"                    int saida_ty,\n"
/*527*/		"\n"
/*528*/		"                    int passox,\n"
/*529*/		"                    int passoy,\n"
/*530*/		"\n"
/*531*/		"                    int largx,\n"
/*532*/		"                    int largy,\n"
/*533*/		"                    int k0) {\n"
/*534*/		"	int k = get_global_id(0) + k0;\n"
/*535*/		"	int m, n, z, l;\n"
/*536*/		"	KTensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*537*/		"	REAL soma = 0,aux;\n"
/*538*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*539*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*540*/		"			aux = entrada[KTensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]\n"
/*541*/		"			        * ds[KTensorMap(i, j, l, saida_tx, saida_ty)];\n"
/*542*/		"			//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*543*/		"			soma += aux;\n"
/*544*/		"		}\n"
/*545*/		"	}\n"
/*546*/		"	gradFiltro[k] = soma;\n"
/*547*/		"}\n"
/*548*/		"\n"
/*549*/		"/**\n"
/*550*/		" * equacao a ser implementada\n"
/*551*/		" * x = s*p + m*w\n"
/*552*/		" * onde:\n"
/*553*/		" * 	x é da entrada \n"
/*554*/		" * 	s é da saida\n"
/*555*/		" * 	m é do filtro\n"
/*556*/		" * 	s = (x - m*w)/p\n"
/*557*/		" */\n"
/*558*/		"kV convncCalcGrads(Vector filtro,\n"
/*559*/		"                   Vector entrada,\n"
/*560*/		"                   Vector gradEntrada,\n"
/*561*/		"                   Vector gradNext,\n"
/*562*/		"\n"
/*563*/		"                   int passox,\n"
/*564*/		"                   int passoy,\n"
/*565*/		"                   int largx,\n"
/*566*/		"                   int largy,\n"
/*567*/		"\n"
/*568*/		"                   int entradatx,\n"
/*569*/		"                   int entradaty,\n"
/*570*/		"                   int saidatx,\n"
/*571*/		"                   int saidaty,\n"
/*572*/		"\n"
/*573*/		"                   int fx,\n"
/*574*/		"                   int fy,\n"
/*575*/		"                   int fz,\n"
/*576*/		"                   int numFilters,\n"
/*577*/		"\n"
/*578*/		"                   int k0) {\n"
/*579*/		"	int k = get_global_id(0) + k0;\n"
/*580*/		"	int x, y, z;\n"
/*581*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*582*/		"	Range range_filtro ;\n"
/*583*/		"	range_filtro.min.x = 0;\n"
/*584*/		"	if ((entradatx - x - (fx - 1) * largx) < 0) {\n"
/*585*/		"		range_filtro.min.x = -entradatx + x + fx;\n"
/*586*/		"	}\n"
/*587*/		"	range_filtro.max.x = fx - 1;\n"
/*588*/		"	if (x - (fx - 1) * largx < 0) {\n"
/*589*/		"		range_filtro.max.x = x / largx;\n"
/*590*/		"	}\n"
/*591*/		"	range_filtro.min.y = 0;\n"
/*592*/		"	if ((entradaty - y - (fy - 1) * largy) < 0) {\n"
/*593*/		"		range_filtro.min.y = -entradaty + y + fy;\n"
/*594*/		"	}\n"
/*595*/		"	range_filtro.max.y = fy - 1;\n"
/*596*/		"	if (y - (fy - 1) * largy < 0) {\n"
/*597*/		"		range_filtro.max.y = y / largy;\n"
/*598*/		"	}\n"
/*599*/		"	int sx, sy;\n"
/*600*/		"	REAL somaErro = 0,aux, pesoAplicado = 0;\n"
/*601*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*602*/		"		sx = (x - m * largx) / passox;\n"
/*603*/		"		if (sx * passox + m * largx != x)continue;\n"
/*604*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*605*/		"			sy = (y - n * largy) / passox;\n"
/*606*/		"			if (sy * passoy + n * largy != y)continue;\n"
/*607*/		"			for (int l = 0; l < fz; l++) {\n"
/*608*/		"				pesoAplicado = filtro[KTensorMap4D(m, n, z, l, fx, fy, fz)];\n"
/*609*/		"				aux = pesoAplicado * gradNext[KTensorMap(sx, sy, l, saidatx, saidaty)];\n"
/*610*/		"				//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*611*/		"				somaErro +=aux;\n"
/*612*/		"			}\n"
/*613*/		"		}\n"
/*614*/		"	}\n"
/*615*/		"	gradEntrada[k] = somaErro;\n"
/*616*/		"}\n"
/*617*/		"\n"
/*618*/		"\n"
/*619*/		"//dropout.h\n"
/*620*/		"#define MAX_INT_DP  ((1UL << 31) - 1)\n"
/*621*/		"long randoml(unsigned long seed,unsigned long id) {\n"
/*622*/		"	seed += id;\n"
/*623*/		"	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;\n"
/*624*/		"}\n"
/*625*/		"\n"
/*626*/		"REAL randomD(unsigned long seed,unsigned long id) {\n"
/*627*/		"	return (REAL) randoml(seed, id) / (REAL) MAX_INT_DP;\n"
/*628*/		"}\n"
/*629*/		"\n"
/*630*/		"kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
/*631*/		"			 REAL pativa, int k0) {\n"
/*632*/		"	int i = get_global_id(0) + k0;\n"
/*633*/		"//	printf(\"kernel %lf %lf %g %g\\n\",randomD(seed, i),pativa,(REAL)(seed +i),(REAL)MAX_INT_DP);\n"
/*634*/		"	char teste = (char) (randomD(seed, i) <= pativa);\n"
/*635*/		"	hitmap[i] = teste;\n"
/*636*/		"	saida[i] = teste * entrada[i];\n"
/*637*/		"}\n"
/*638*/		"\n"
/*639*/		"\n"
/*640*/		"kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {\n"
/*641*/		"	int i = get_global_id(0) + k0;\n"
/*642*/		"	gradentrada[i] = hitmap[i] * gradnext[i];\n"
/*643*/		"}\n"
/*644*/		"\n"
/*645*/		"//fullconnect.h\n"
/*646*/		"\n"
/*647*/		"\n"
/*648*/		"kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,\n"
/*649*/		"			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {\n"
/*650*/		"	int m = get_global_id(0) + k0;\n"
/*651*/		"	REAL valorEntrada = 0;\n"
/*652*/		"	int n;\n"
/*653*/		"	for (n = 0; n < pesosy; n++) {\n"
/*654*/		"		valorEntrada += entrada[n] * pesos[KTensorMap(m, n, 0, pesosx, pesosy)];\n"
/*655*/		"	}\n"
/*656*/		"	z[m] = valorEntrada;\n"
/*657*/		"	saida[m] = func(funcaoativacao, valorEntrada);\n"
/*658*/		"}\n"
/*659*/		"\n"
/*660*/		"kV fullfixweight(Vector a,\n"
/*661*/		"			  Vector pesos,\n"
/*662*/		"			  Vector dw,\n"
/*663*/		"			  Vector dz,\n"
/*664*/		"			  REAL hitlearn,\n"
/*665*/		"			  REAL decaimentoDePeso,\n"
/*666*/		"			  REAL momento,\n"
/*667*/		"			  int pesosy,\n"
/*668*/		"			  int k0) {\n"
/*669*/		"	int k = get_global_id(0) + k0;\n"
/*670*/		"	int m, n;\n"
/*671*/		"	m = k / pesosy;\n"
/*672*/		"	n = k % pesosy;\n"
/*673*/		"	dw[k] = dz[m] * a[n] + dw[k] * momento;\n"
/*674*/		"	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);\n"
/*675*/		"}\n"
/*676*/		"\n"
/*677*/		"kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {\n"
/*678*/		"	int m = get_global_id(0) + k0;\n"
/*679*/		"	dz[m] = ds[m] * func(dfa, z[m]);\n"
/*680*/		"}\n"
/*681*/		"\n"
/*682*/		"kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,\n"
/*683*/		"				  int k0) {\n"
/*684*/		"	int m = get_global_id(0) + k0;\n"
/*685*/		"	REAL soma = 0;\n"
/*686*/		"	for (int n = 0; n < pesosx; ++n) {\n"
/*687*/		"		soma += dz[n] * pesos[KTensorMap(n, m, 0, pesosx, pesosy)];\n"
/*688*/		"	}\n"
/*689*/		"	da[m] = soma;\n"
/*690*/		"}\n"
/*691*/		"\n"
/*692*/		"//padding.h\n"
/*693*/		"kV paddingfeed(Vector in,Vector out,\n"
/*694*/		"			   int txi,int tyi,\n"
/*695*/		"			   int txo,int tyo,\n"
/*696*/		"			   int t, int l ,\n"
/*697*/		"			   int k0){\n"
/*698*/		"	int k = get_global_id(0) + k0;\n"
/*699*/		"	int x, y, z;\n"
/*700*/		"	KTensorRemap(k, x, y, z, txi, tyi)\n"
/*701*/		"	int s = KTensorMap(x+t,y+l,z,txo,tyo);\n"
/*702*/		"	out[s] = in[k];\n"
/*703*/		"}\n"
/*704*/		"kV paddingBack(Vector gradNext,Vector gradin,\n"
/*705*/		"			   int txi,int tyi,\n"
/*706*/		"			   int txo,int tyo,\n"
/*707*/		"			   int t, int l , int k0){\n"
/*708*/		"	int k = get_global_id(0) + k0;\n"
/*709*/		"	int x, y, z;\n"
/*710*/		"	KTensorRemap(k, x, y, z, txi, tyi)\n"
/*711*/		"	int s = KTensorMap(x+t,y+l,z,txo,tyo);\n"
/*712*/		"	gradin[k] = gradNext[s];\n"
/*713*/		"}\n"
/*714*/		"//pool.h\n"
/*715*/		"kV poolativa(Vector entrada, Vector saida,\n"
/*716*/		"			 int passox,int passoy,\n"
/*717*/		"			 int filtrox,int filtroy,\n"
/*718*/		"			 int saidatx, int saidaty,\n"
/*719*/		"			 int entradatx, int entradaty, int k0) {\n"
/*720*/		"	int k = get_global_id(0) + k0;\n"
/*721*/		"	int x, y, z;\n"
/*722*/		"	KTensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*723*/		"\n"
/*724*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*725*/		"	REAL mval, v;\n"
/*726*/		"	mval = -DBL_MAX;\n"
/*727*/		"	for (int i = 0; i < filtrox; ++i) {\n"
/*728*/		"		for (int j = 0; j < filtroy; ++j) {\n"
/*729*/		"			v = entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*730*/		"			if (v > mval)\n"
/*731*/		"				mval = v;\n"
/*732*/		"		}\n"
/*733*/		"	}\n"
/*734*/		"	saida[k] = mval;\n"
/*735*/		"}\n"
/*736*/		"\n"
/*737*/		"\n"
/*738*/		"kV poolCalcGrads(Vector entrada, Vector gradEntrada,\n"
/*739*/		"				 Vector gradNext, Vector saida,\n"
/*740*/		"				 int fx, int fy, int px, int py,\n"
/*741*/		"				 int entradatx, int entradaty,\n"
/*742*/		"				 int saidatx, int saidaty,\n"
/*743*/		"				 int k0) {\n"
/*744*/		"	int k = get_global_id(0) + k0;\n"
/*745*/		"	int x, y, z;\n"
/*746*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*747*/		"	Range range_filtro;\n"
/*748*/		"	if (x + fx > entradatx) {\n"
/*749*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*750*/		"	}\n"
/*751*/		"	range_filtro.max.x = fx - 1;\n"
/*752*/		"	if (x - fx + 1 < 0) {\n"
/*753*/		"		range_filtro.max.x = x;\n"
/*754*/		"	}\n"
/*755*/		"	range_filtro.min.y = 0;\n"
/*756*/		"	if (y + fy > entradaty) {\n"
/*757*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*758*/		"	}\n"
/*759*/		"	range_filtro.max.y = fy - 1;\n"
/*760*/		"	if (y - fy + 1 < 0) {\n"
/*761*/		"		range_filtro.max.y = y;\n"
/*762*/		"	}\n"
/*763*/		"	int i, j;//saida\n"
/*764*/		"	gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] =0;\n"
/*765*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*766*/		"		i = (x - m) / px;\n"
/*767*/		"		if (i * px + m != x)continue;\n"
/*768*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*769*/		"			j = (y - n) / py;\n"
/*770*/		"			if (j * py + n != y)continue;\n"
/*771*/		"			if (entrada[KTensorMap(x, y, z, entradatx, entradaty)] ==\n"
/*772*/		"				saida[KTensorMap(i, j, z, saidatx, saidaty)]) {\n"
/*773*/		"				gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] =\n"
/*774*/		"						gradNext[KTensorMap(i, j, z, saidatx, saidaty)];\n"
/*775*/		"				return;\n"
/*776*/		"			}\n"
/*777*/		"		}\n"
/*778*/		"	}\n"
/*779*/		"\n"
/*780*/		"}\n"
/*781*/		"\n"
/*782*/		"\n"
/*783*/		"//poolav.h\n"
/*784*/		"kV PoolAvativa(Vector entrada, Vector saida,\n"
/*785*/		"			   int passox,int passoy,\n"
/*786*/		"			   int fx,int fy,\n"
/*787*/		"			   int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*788*/		"	int k = get_global_id(0) + k0;\n"
/*789*/		"	int x, y, z;\n"
/*790*/		"	KTensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*791*/		"\n"
/*792*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*793*/		"	REAL soma = 0, v;\n"
/*794*/		"\n"
/*795*/		"	for (int i = 0; i < fx; ++i) {\n"
/*796*/		"		for (int j = 0; j < fy; ++j) {\n"
/*797*/		"			soma += entrada[KTensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*798*/		"		}\n"
/*799*/		"	}\n"
/*800*/		"	saida[k] = soma / (fx * fy);\n"
/*801*/		"}\n"
/*802*/		"\n"
/*803*/		"\n"
/*804*/		"kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*805*/		"                   int px, int py,  int fx, int fy,\n"
/*806*/		"				   int entradatx, int entradaty, int saidatx, int saidaty,\n"
/*807*/		"				   int k0) {\n"
/*808*/		"	int k = get_global_id(0) + k0;\n"
/*809*/		"	int x, y, z;\n"
/*810*/		"	KTensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*811*/		"	Range range_filtro;\n"
/*812*/		"	range_filtro.min.x = 0;\n"
/*813*/		"	if (x + fx > entradatx) {\n"
/*814*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*815*/		"	}\n"
/*816*/		"	range_filtro.max.x = fx - 1;\n"
/*817*/		"	if (x - fx + 1 < 0) {\n"
/*818*/		"		range_filtro.max.x = x;\n"
/*819*/		"	}\n"
/*820*/		"	range_filtro.min.y = 0;\n"
/*821*/		"	if (y + fy > entradaty) {\n"
/*822*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*823*/		"	}\n"
/*824*/		"	range_filtro.max.y = fy - 1;\n"
/*825*/		"	if (y - fy + 1 < 0) {\n"
/*826*/		"		range_filtro.max.y = y;\n"
/*827*/		"	}\n"
/*828*/		"	int i, j;//saida\n"
/*829*/		"	REAL soma = 0;\n"
/*830*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*831*/		"		i = (x - m) / px;\n"
/*832*/		"		if (i * px + m != x)continue;\n"
/*833*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*834*/		"			j = (y - n) / py;\n"
/*835*/		"			if (j * py + n != y)continue;\n"
/*836*/		"			soma += gradNext[KTensorMap(i, j, z, saidatx, saidaty)];\n"
/*837*/		"		}\n"
/*838*/		"	}\n"
/*839*/		"	gradEntrada[KTensorMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);\n"
/*840*/		"\n"
/*841*/		"}\n"
/*842*/		"\n"
/*843*/		"\n"
/*844*/		"//prelu.h\n"
/*845*/		"kV preluativa(Vector entrada, Vector saida, Vector A, int k0) {\n"
/*846*/		"	int k = get_global_id(0) + k0;\n"
/*847*/		"	REAL v = entrada[k];\n"
/*848*/		"	if (v < 0)\n"
/*849*/		"		v = v * A[k];\n"
/*850*/		"	saida[k] = v;\n"
/*851*/		"}\n"
/*852*/		"\n"
/*853*/		"kV prelucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, Vector A, Vector dA,\n"
/*854*/		"				 int learn,REAL hitlearn, REAL momento,\n"
/*855*/		"				 REAL decaimento,\n"
/*856*/		"				 int k0) {\n"
/*857*/		"	int k = get_global_id(0) + k0;\n"
/*858*/		"	REAL v = entrada[k];\n"
/*859*/		"	if (v < 0) {\n"
/*860*/		"		gradentrada[k] = gradnext[k] * A[k];\n"
/*861*/		"		dA[k] = gradnext[k] + momento * dA[k];\n"
/*862*/		"	} else {\n"
/*863*/		"		gradentrada[k] = gradnext[k];\n"
/*864*/		"		dA[k] = momento * dA[k];\n"
/*865*/		"	}\n"
/*866*/		"	if (learn)\n"
/*867*/		"		A[k] = A[k] - hitlearn * (dA[k] + A[k] * decaimento);\n"
/*868*/		"}\n"
/*869*/		"\n"
/*870*/		"//relu.h\n"
/*871*/		"kV reluativa(Vector entrada, Vector saida, REAL menor, REAL maior, int k0) {\n"
/*872*/		"	int k = get_global_id(0) + k0;\n"
/*873*/		"	saida[k] = entrada[k] < 0.0 ? (entrada[k] * menor) : (entrada[k]* maior);\n"
/*874*/		"}\n"
/*875*/		"\n"
/*876*/		"kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, REAL menor, REAL maior, int k0) {\n"
/*877*/		"	int k = get_global_id(0) + k0;\n"
/*878*/		"	gradentrada[k] = entrada[k] < 0.0 ? (menor*gradnext[k]) : (maior*gradnext[k]);\n"
/*879*/		"}\n"
/*880*/		"\n"
/*881*/		"//softmax.h\n"
/*882*/		"kV SoftMaxativa1(Vector entrada, Vector exponent,\n"
/*883*/		"				 int k0) {\n"
/*884*/		"	int k = get_global_id(0) + k0;\n"
/*885*/		"	exponent[k] = exp(entrada[k]);\n"
/*886*/		"}\n"
/*887*/		"\n"
/*888*/		"\n"
/*889*/		"\n"
/*890*/		"kV SoftMaxativa2(Vector exponent, Vector soma,\n"
/*891*/		"				 int saidatx, int saidaty, int k0) {\n"
/*892*/		"	int z = get_global_id(0) + k0;\n"
/*893*/		"	int x, y;\n"
/*894*/		"	int d;\n"
/*895*/		"	REAL sum;\n"
/*896*/		"	for (x = 0; x < saidatx; x++)\n"
/*897*/		"		for (y = 0; y < saidaty; y++) {\n"
/*898*/		"			d = KTensorMap(x, y, z, saidatx, saidaty);\n"
/*899*/		"			sum += exponent[d];\n"
/*900*/		"		}\n"
/*901*/		"	soma[z] = sum;\n"
/*902*/		"}\n"
/*903*/		"\n"
/*904*/		"kV SoftMaxativa3(Vector exponet, Vector soma, Vector saida,\n"
/*905*/		"				 int saidatx, int saidaty, int k0) {\n"
/*906*/		"	int k = get_global_id(0) + k0;\n"
/*907*/		"	int x, y, z;\n"
/*908*/		"	KTensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*909*/		"	saida[k] = exponet[KTensorMap(x, y, z, saidatx, saidaty)] / soma[z];\n"
/*910*/		"}\n"
/*911*/		"kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*912*/		"	int k = get_global_id(0) + k0;\n"
/*913*/		"	REAL xi = entrada[k];\n"
/*914*/		"	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];\n"
/*915*/		"}\n"
/*916*/		"\n"
/*917*/		"\n"
/*918*/		"#endif //GAB_KERNELS_OPENCL_H\n"
;
const char *getInternalDefaultKernel() {
	return __default_kernel__;
}