//
// Created by hslhe on 14/08/2021.
//
#include "utils/defaultkernel.h"
const char __default_kernel__[] = 
/*1*/		"#ifndef GAB_KERNELS_OPENCL_H\n"
/*2*/		"#define GAB_KERNELS_OPENCL_H\n"
/*3*/		"//utils.h\n"
/*4*/		"// Created by Xx220xX on 10/05/2020.\n"
/*5*/		"\n"
/*6*/		"#define Vector __global double *\n"
/*7*/		"\n"
/*8*/		"#define kV __kernel void\n"
/*9*/		"\n"
/*10*/		"#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))\n"
/*11*/		"\n"
/*12*/		"#define TensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))\n"
/*13*/		"\n"
/*14*/		"#define TensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\\\n"
/*15*/		"_y_ = total%ty      ;                                        \\\n"
/*16*/		"_x_ = (total - _y_)%(ty*tx)/ty ;                             \\\n"
/*17*/		"_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \\\n"
/*18*/		"_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);\n"
/*19*/		"\n"
/*20*/		"\n"
/*21*/		"#define TensorRemap(total, _x_, _y_, _z_, tx, ty)\\\n"
/*22*/		"_y_ = total % ty;\\\n"
/*23*/		"_x_ = ((total - _y_) % (ty * tx)) / ty;\\\n"
/*24*/		"_z_ = (k - _x_ * ty - _y_) / (tx * ty);\n"
/*25*/		"\n"
/*26*/		"#define TensorRemap2D(total, x, y, ty)\\\n"
/*27*/		"y = total % ty;\\\n"
/*28*/		"x = total/ ty;\n"
/*29*/		"\n"
/*30*/		"typedef struct {\n"
/*31*/		"	int x, y, z;\n"
/*32*/		"} Ponto3d;\n"
/*33*/		"\n"
/*34*/		"typedef struct {\n"
/*35*/		"	Ponto3d min, max;\n"
/*36*/		"} Range;\n"
/*37*/		"\n"
/*38*/		"//bathnorm.h\n"
/*39*/		"\n"
/*40*/		"// achar a media\n"
/*41*/		"kV BatchNormMedia(Vector entrada, Vector media,\n"
/*42*/		"                  int entradatx, int entradaty, int k0) {\n"
/*43*/		"	int z = get_global_id(0) + k0;\n"
/*44*/		"	int x, y;\n"
/*45*/		"	double m = 0;\n"
/*46*/		"	for (x = 0; x < entradatx; x++) {\n"
/*47*/		"		for (y = 0; y < entradaty; y++) {\n"
/*48*/		"			m += entrada[TensorMap(x, y, z, entradatx, entradaty)];\n"
/*49*/		"		}\n"
/*50*/		"	}\n"
/*51*/		"	media[z] = m / (double) (entradatx * entradaty);\n"
/*52*/		"}\n"
/*53*/		"\n"
/*54*/		"// achar a diferenca\n"
/*55*/		"kV BatchNormDiferenca(Vector entrada, Vector media,\n"
/*56*/		"                      Vector diferenca,\n"
/*57*/		"                      Vector diferencaquad,\n"
/*58*/		"                      int entradatx, int entradaty, int k0) {\n"
/*59*/		"	int x, y, z;\n"
/*60*/		"	int k = get_global_id(0) + k0;\n"
/*61*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*62*/		"	diferenca[k] = entrada[k] - media[z];\n"
/*63*/		"	diferencaquad[k] = diferenca[k] * diferenca[k];\n"
/*64*/		"}\n"
/*65*/		"\n"
/*66*/		"kV BatchNormVariance(Vector dif, Vector difQuad,\n"
/*67*/		"                     Vector sumdiferenca, Vector variancia,\n"
/*68*/		"                     double episolon, int diftx, int difty,\n"
/*69*/		"                     int k0) {\n"
/*70*/		"	int z = get_global_id(0) + k0;\n"
/*71*/		"	double sum = 0;\n"
/*72*/		"	double sumdif = 0;\n"
/*73*/		"	for (int x = 0; x < diftx; x++) {\n"
/*74*/		"		for (int y = 0; y < difty; y++) {\n"
/*75*/		"			sum += difQuad[TensorMap(x, y, z, diftx, difty)];\n"
/*76*/		"			sumdif += dif[TensorMap(x, y, z, diftx, difty)];\n"
/*77*/		"		}\n"
/*78*/		"	}\n"
/*79*/		"	sumdiferenca[z] = sumdif;\n"
/*80*/		"	variancia[z] = sqrt(sum / (difty * diftx) + episolon);\n"
/*81*/		"}\n"
/*82*/		"\n"
/*83*/		"// normaliza\n"
/*84*/		"kV BatchNormNormaliza(Vector saida,\n"
/*85*/		"                      Vector norma,\n"
/*86*/		"                      Vector diferenca,\n"
/*87*/		"                      Vector variancia,\n"
/*88*/		"                      Vector Y,\n"
/*89*/		"                      Vector B,\n"
/*90*/		"                      int diferencatx, int diferencaty, int k0) {\n"
/*91*/		"	int x, y, z;\n"
/*92*/		"	int k = get_global_id(0) + k0;\n"
/*93*/		"	TensorRemap(k, x, y, z, diferencatx, diferencaty)\n"
/*94*/		"	norma[k] = diferenca[k] / variancia[z];\n"
/*95*/		"	saida[k] = norma[k] * Y[z] + B[z];\n"
/*96*/		"}\n"
/*97*/		"\n"
/*98*/		"\n"
/*99*/		"kV BatchNormaCalcGrad1(Vector gradIn,\n"
/*100*/		"                       Vector gradNext,\n"
/*101*/		"                       Vector variancia,\n"
/*102*/		"                       Vector media,\n"
/*103*/		"                       Vector Y,\n"
/*104*/		"\n"
/*105*/		"                       Vector somaDif,\n"
/*106*/		"                       Vector entrada,\n"
/*107*/		"                       int entradatx,\n"
/*108*/		"                       int entradaty,\n"
/*109*/		"                       int k0) {\n"
/*110*/		"	int x, y, z;\n"
/*111*/		"	int k = get_global_id(0) + k0;\n"
/*112*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*113*/		"	double M = entradatx * entradaty;\n"
/*114*/		"	double dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);\n"
/*115*/		"	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);\n"
/*116*/		"\n"
/*117*/		"	double didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;\n"
/*118*/		"	didx = didx / (variancia[z] * variancia[z]);\n"
/*119*/		"	didx = didx * gradNext[k];\n"
/*120*/		"	gradIn[k] = didx * Y[z];\n"
/*121*/		"}\n"
/*122*/		"\n"
/*123*/		"kV BatchNormaCalcGrad2(Vector gradNext,\n"
/*124*/		"                       Vector norma,\n"
/*125*/		"                       Vector gradY,\n"
/*126*/		"                       Vector gradB,\n"
/*127*/		"                       int entradatx,\n"
/*128*/		"                       int entradaty,\n"
/*129*/		"                       int k0) {\n"
/*130*/		"	int z = get_global_id(0) + k0;\n"
/*131*/		"	double sumY = 0;\n"
/*132*/		"	double sumB = 0;\n"
/*133*/		"	int k;\n"
/*134*/		"	for (int x = 0; x < entradatx; ++x) {\n"
/*135*/		"		for (int y = 0; y < entradaty; ++y) {\n"
/*136*/		"			k = TensorMap(x, y, z, entradatx, entradaty);\n"
/*137*/		"			sumY += gradNext[k];\n"
/*138*/		"			sumB += gradNext[k] * norma[k];\n"
/*139*/		"		}\n"
/*140*/		"	}\n"
/*141*/		"	gradB[z] = sumB;\n"
/*142*/		"	gradY[z] = sumY;\n"
/*143*/		"}\n"
/*144*/		"\n"
/*145*/		"\n"
/*146*/		"kV batchNormCorrigePeso(Vector gradY,\n"
/*147*/		"                        Vector gradB,\n"
/*148*/		"                        Vector Y,\n"
/*149*/		"                        Vector B,\n"
/*150*/		"                        double hitlearn,\n"
/*151*/		"                        int k0) {\n"
/*152*/		"	int z = get_global_id(0) + k0;\n"
/*153*/		"	B[z] = B[z] - gradB[z] * hitlearn;\n"
/*154*/		"	Y[z] = Y[z] - gradY[z] * hitlearn;\n"
/*155*/		"}\n"
/*156*/		"//cnnutils.h\n"
/*157*/		"//\n"
/*158*/		"// Created by Henrique on 22-Jul-21.\n"
/*159*/		"//\n"
/*160*/		"\n"
/*161*/		"\n"
/*162*/		"kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {\n"
/*163*/		"	int k = get_global_id(0) + k0;\n"
/*164*/		"	int i, j, z;\n"
/*165*/		"	TensorRemap(k, i, j, z, vx, vy)\n"
/*166*/		"	imi = imi + i;\n"
/*167*/		"	int imj = j + z * vy + z;\n"
/*168*/		"	out[imi * imy + imj] = ((int) v[k]) & 0xff;\n"
/*169*/		"}\n"
/*170*/		"\n"
/*171*/		"\n"
/*172*/		"kV\n"
/*173*/		"normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,\n"
/*174*/		"				int k0) {\n"
/*175*/		"	int k = get_global_id(0) + k0;\n"
/*176*/		"	saida[k] = (input[k] + somador) * multiplicador - subtrator;\n"
/*177*/		"}\n"
/*178*/		"\n"
/*179*/		"\n"
/*180*/		"kV subKernel(Vector grad, Vector saida, Vector target, int k0) {\n"
/*181*/		"	int k = get_global_id(0) + k0;\n"
/*182*/		"	grad[k] = saida[k] - target[k];\n"
/*183*/		"}\n"
/*184*/		"\n"
/*185*/		"kV divKernel(Vector v, double value, int k0) {\n"
/*186*/		"	int k = get_global_id(0) + k0;\n"
/*187*/		"	v[k] = v[k] / value;\n"
/*188*/		"}\n"
/*189*/		"\n"
/*190*/		"kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {\n"
/*191*/		"	int k = get_global_id(0) + k0;\n"
/*192*/		"	v[k] = ((double) src[k]) / value;\n"
/*193*/		"}\n"
/*194*/		"\n"
/*195*/		"kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {\n"
/*196*/		"	int k = get_global_id(0) + k0;\n"
/*197*/		"\n"
/*198*/		"	int d;\n"
/*199*/		"//	printf(\"%d %d \",k,ints[k]);\n"
/*200*/		"	for (int j = 0; j < noptiobs; j++) {\n"
/*201*/		"		d = TensorMap4D(0, j, 0, k, 1, noptiobs, 1);\n"
/*202*/		"//		printf(\"%d \",d);\n"
/*203*/		"		v[d] = (double) (j == ints[k]);\n"
/*204*/		"	}\n"
/*205*/		"//	printf(\"\\n\");\n"
/*206*/		"}\n"
/*207*/		"\n"
/*208*/		"\n"
/*209*/		"\n"
/*210*/		"\n"
/*211*/		"//conv.h\n"
/*212*/		"kV convSum(Vector filtro, Vector entrada, Vector saida,\n"
/*213*/		"           int passox, int passoy,\n"
/*214*/		"           int saidatx, int saidaty,\n"
/*215*/		"           int entradatx, int entradaty,\n"
/*216*/		"           int fx, int fy, int fz, int k0) {\n"
/*217*/		"	int k = get_global_id(0) + k0;\n"
/*218*/		"	int x, y, filtrok;\n"
/*219*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*220*/		"	double sum = 0, f = 0, v = 0;\n"
/*221*/		"	int lf = 0, le = 0;\n"
/*222*/		"	for (int m = 0; m < fx; m++) {\n"
/*223*/		"		for (int n = 0; n < fy; n++) {\n"
/*224*/		"			for (int z = 0; z < fz; z++) {\n"
/*225*/		"				lf = TensorMap4D(m, n, z, filtrok, fx, fy, fz);\n"
/*226*/		"				le = TensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);\n"
/*227*/		"				f = filtro[lf];\n"
/*228*/		"				v = entrada[le];\n"
/*229*/		"				sum += f * v;\n"
/*230*/		"			}\n"
/*231*/		"		}\n"
/*232*/		"	}\n"
/*233*/		"	saida[k] = sum;\n"
/*234*/		"}\n"
/*235*/		"\n"
/*236*/		"\n"
/*237*/		"kV convCalcGradAndFixWeight(Vector filtros, Vector ds,\n"
/*238*/		"                            Vector entrada, Vector gradFiltro,\n"
/*239*/		"                            int fx, int fy, int fz,\n"
/*240*/		"                            int entrada_tx, int entrada_ty,\n"
/*241*/		"                            int saida_tx, int saida_ty,\n"
/*242*/		"                            int passox, int passoy,\n"
/*243*/		"                            double hitLearn, double momento, double weightDecay,\n"
/*244*/		"                            int k0) {\n"
/*245*/		"	int k = get_global_id(0) + k0;\n"
/*246*/		"	int m, n, z, l;\n"
/*247*/		"	TensorRemap4D(k, m, n, z, l, fx, fy, fz)\n"
/*248*/		"	double soma = 0;\n"
/*249*/		"	int le, ls;\n"
/*250*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*251*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*252*/		"			le = TensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);\n"
/*253*/		"			ls = TensorMap(i, j, l, saida_tx, saida_ty);\n"
/*254*/		"			soma += entrada[le]\n"
/*255*/		"			        * ds[ls];\n"
/*256*/		"		}\n"
/*257*/		"	}\n"
/*258*/		"	double dw = soma + gradFiltro[k] * momento;\n"
/*259*/		"	double w = filtros[k];\n"
/*260*/		"	filtros[k] = w - hitLearn * (dw + w * weightDecay);\n"
/*261*/		"	gradFiltro[k] = dw;\n"
/*262*/		"}\n"
/*263*/		"\n"
/*264*/		"kV convCalcGradIn(Vector filtro, Vector gradEntrada, Vector gradNext,\n"
/*265*/		"                  int fx, int fy, int fz,\n"
/*266*/		"                  int passox, int passoy,\n"
/*267*/		"                  int entradatx, int entradaty,\n"
/*268*/		"                  int saidatx, int saidaty, int saidatz,\n"
/*269*/		"                  int k0) {\n"
/*270*/		"	int k = get_global_id(0) + k0;\n"
/*271*/		"	int x, y, z;\n"
/*272*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*273*/		"\n"
/*274*/		"	Range range_filtro;\n"
/*275*/		"	range_filtro.min.x = 0;\n"
/*276*/		"	if (x + fx > entradatx) {\n"
/*277*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*278*/		"	}\n"
/*279*/		"	range_filtro.max.x = fx - 1;\n"
/*280*/		"	if (x - fx + 1 < 0) {\n"
/*281*/		"		range_filtro.max.x = x;\n"
/*282*/		"	}\n"
/*283*/		"	range_filtro.min.y = 0;\n"
/*284*/		"	if (y + fy > entradaty) {\n"
/*285*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*286*/		"	}\n"
/*287*/		"	range_filtro.max.y = fy - 1;\n"
/*288*/		"	if (y - fy + 1 < 0) {\n"
/*289*/		"		range_filtro.max.y = y;\n"
/*290*/		"	}\n"
/*291*/		"	double somaErro = 0, pesoAplicado = 0;\n"
/*292*/		"	int i, j;\n"
/*293*/		"	int lf, ls;\n"
/*294*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*295*/		"		i = (x - m) / passox;\n"
/*296*/		"		if (i * passox + m != x) continue;\n"
/*297*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*298*/		"			j = (y - n) / passoy;\n"
/*299*/		"			if (j * passoy + n != y) continue;\n"
/*300*/		"			for (int w = 0; w < saidatz; w++) {\n"
/*301*/		"				lf = TensorMap4D(m, n, z, w, fx, fy, fz);\n"
/*302*/		"				ls = TensorMap(i, j, w, saidatx, saidaty);\n"
/*303*/		"				pesoAplicado = filtro[lf];\n"
/*304*/		"				somaErro += pesoAplicado * gradNext[ls];\n"
/*305*/		"			}\n"
/*306*/		"		}\n"
/*307*/		"	}\n"
/*308*/		"	gradEntrada[k] = somaErro;\n"
/*309*/		"}\n"
/*310*/		"\n"
/*311*/		"\n"
/*312*/		"//convNc.h\n"
/*313*/		"//#include\"utils.h\"\n"
/*314*/		"kV convncSum(Vector filtro, Vector entrada, Vector saida,\n"
/*315*/		"             int passox, int passoy, int largx,\n"
/*316*/		"             int largy, int saidatx, int saidaty,\n"
/*317*/		"             int entradatx, int entradaty,int fx, int fy,\n"
/*318*/		"             int entradatz, int k0) {\n"
/*319*/		"	int k = get_global_id(0) + k0;\n"
/*320*/		"	int x, y, filtrok;\n"
/*321*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*322*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*323*/		"	double sum = 0, f, v;\n"
/*324*/		"	for (int i = 0; i < fx; i++)\n"
/*325*/		"		for (int j = 0; j < fy; j++)\n"
/*326*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*327*/		"				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];\n"
/*328*/		"				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];\n"
/*329*/		"\n"
/*330*/		"				sum += f * v;\n"
/*331*/		"			}\n"
/*332*/		"	saida[k] = sum;\n"
/*333*/		"}\n"
/*334*/		"\n"
/*335*/		"kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,\n"
/*336*/		"				   double hitlearn,\n"
/*337*/		"                   double momento, double weightDecay, int k0) {\n"
/*338*/		"	int k = get_global_id(0) + k0;\n"
/*339*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*340*/		"	double w = filtro[k];\n"
/*341*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*342*/		"	gradOld[k] = m;\n"
/*343*/		"}\n"
/*344*/		"\n"
/*345*/		"kV convncCalcFiltro(Vector ds,\n"
/*346*/		"                    Vector entrada,\n"
/*347*/		"                    Vector gradFiltro,\n"
/*348*/		"                    int gradFiltro_tx,\n"
/*349*/		"                    int gradFiltro_ty,\n"
/*350*/		"                    int gradFiltro_tz,\n"
/*351*/		"\n"
/*352*/		"                    int entrada_tx,\n"
/*353*/		"                    int entrada_ty,\n"
/*354*/		"\n"
/*355*/		"                    int saida_tx,\n"
/*356*/		"                    int saida_ty,\n"
/*357*/		"\n"
/*358*/		"                    int passox,\n"
/*359*/		"                    int passoy,\n"
/*360*/		"\n"
/*361*/		"                    int largx,\n"
/*362*/		"                    int largy,\n"
/*363*/		"                    int k0) {\n"
/*364*/		"	int k = get_global_id(0) + k0;\n"
/*365*/		"	int m, n, z, l;\n"
/*366*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*367*/		"	double soma = 0,aux;\n"
/*368*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*369*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*370*/		"			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]\n"
/*371*/		"			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];\n"
/*372*/		"			//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*373*/		"			soma += aux;\n"
/*374*/		"		}\n"
/*375*/		"	}\n"
/*376*/		"	gradFiltro[k] = soma;\n"
/*377*/		"}\n"
/*378*/		"\n"
/*379*/		"/**\n"
/*380*/		" * equacao a ser implementada\n"
/*381*/		" * x = s*p + m*w\n"
/*382*/		" * onde:\n"
/*383*/		" * 	x é da entrada \n"
/*384*/		" * 	s é da saida\n"
/*385*/		" * 	m é do filtro\n"
/*386*/		" * 	s = (x - m*w)/p\n"
/*387*/		" */\n"
/*388*/		"kV convncCalcGrads(Vector filtro,\n"
/*389*/		"                   Vector entrada,\n"
/*390*/		"                   Vector gradEntrada,\n"
/*391*/		"                   Vector gradNext,\n"
/*392*/		"\n"
/*393*/		"                   int passox,\n"
/*394*/		"                   int passoy,\n"
/*395*/		"                   int largx,\n"
/*396*/		"                   int largy,\n"
/*397*/		"\n"
/*398*/		"                   int entradatx,\n"
/*399*/		"                   int entradaty,\n"
/*400*/		"                   int saidatx,\n"
/*401*/		"                   int saidaty,\n"
/*402*/		"\n"
/*403*/		"                   int fx,\n"
/*404*/		"                   int fy,\n"
/*405*/		"                   int fz,\n"
/*406*/		"                   int numFilters,\n"
/*407*/		"\n"
/*408*/		"                   int k0) {\n"
/*409*/		"	int k = get_global_id(0) + k0;\n"
/*410*/		"	int x, y, z;\n"
/*411*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*412*/		"	Range range_filtro ;\n"
/*413*/		"	range_filtro.min.x = 0;\n"
/*414*/		"	if ((entradatx - x - (fx - 1) * largx) < 0) {\n"
/*415*/		"		range_filtro.min.x = -entradatx + x + fx;\n"
/*416*/		"	}\n"
/*417*/		"	range_filtro.max.x = fx - 1;\n"
/*418*/		"	if (x - (fx - 1) * largx < 0) {\n"
/*419*/		"		range_filtro.max.x = x / largx;\n"
/*420*/		"	}\n"
/*421*/		"	range_filtro.min.y = 0;\n"
/*422*/		"	if ((entradaty - y - (fy - 1) * largy) < 0) {\n"
/*423*/		"		range_filtro.min.y = -entradaty + y + fy;\n"
/*424*/		"	}\n"
/*425*/		"	range_filtro.max.y = fy - 1;\n"
/*426*/		"	if (y - (fy - 1) * largy < 0) {\n"
/*427*/		"		range_filtro.max.y = y / largy;\n"
/*428*/		"	}\n"
/*429*/		"	int sx, sy;\n"
/*430*/		"	double somaErro = 0,aux, pesoAplicado = 0;\n"
/*431*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*432*/		"		sx = (x - m * largx) / passox;\n"
/*433*/		"		if (sx * passox + m * largx != x)continue;\n"
/*434*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*435*/		"			sy = (y - n * largy) / passox;\n"
/*436*/		"			if (sy * passoy + n * largy != y)continue;\n"
/*437*/		"			for (int l = 0; l < fz; l++) {\n"
/*438*/		"				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];\n"
/*439*/		"				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];\n"
/*440*/		"				//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*441*/		"				somaErro +=aux;\n"
/*442*/		"			}\n"
/*443*/		"		}\n"
/*444*/		"	}\n"
/*445*/		"	gradEntrada[k] = somaErro;\n"
/*446*/		"}\n"
/*447*/		"\n"
/*448*/		"\n"
/*449*/		"//dropout.h\n"
/*450*/		"#define MAX_INT_DP  ((1UL << 31) - 1)\n"
/*451*/		"long randoml(unsigned long seed,unsigned long id) {\n"
/*452*/		"	seed += id;\n"
/*453*/		"	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;\n"
/*454*/		"}\n"
/*455*/		"\n"
/*456*/		"double randomD(unsigned long seed,unsigned long id) {\n"
/*457*/		"	return (double) randoml(seed, id) / (double) MAX_INT_DP;\n"
/*458*/		"}\n"
/*459*/		"\n"
/*460*/		"kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
/*461*/		"			 double pativa, int k0) {\n"
/*462*/		"	int i = get_global_id(0) + k0;\n"
/*463*/		"//	printf(\"kernel %lf %lf %g %g\\n\",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);\n"
/*464*/		"	char teste = (char) (randomD(seed, i) <= pativa);\n"
/*465*/		"	hitmap[i] = teste;\n"
/*466*/		"	saida[i] = teste * entrada[i];\n"
/*467*/		"}\n"
/*468*/		"\n"
/*469*/		"\n"
/*470*/		"kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {\n"
/*471*/		"	int i = get_global_id(0) + k0;\n"
/*472*/		"	gradentrada[i] = hitmap[i] * gradnext[i];\n"
/*473*/		"}\n"
/*474*/		"\n"
/*475*/		"//fullconnect.h\n"
/*476*/		"double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }\n"
/*477*/		"\n"
/*478*/		"double difsigmoid(double x) {\n"
/*479*/		"	double tmp = sigmoid(x);\n"
/*480*/		"	return tmp * (1.0 - tmp);\n"
/*481*/		"}\n"
/*482*/		"\n"
/*483*/		"double tanghG(double x) { return tanh(x); }\n"
/*484*/		"\n"
/*485*/		"double diftanhG(double x) {\n"
/*486*/		"	double tmp = tanh(x);\n"
/*487*/		"	return (1.0 - tmp * tmp);\n"
/*488*/		"}\n"
/*489*/		"\n"
/*490*/		"double relu(double x) { return x > 0 ? x : 0.0; }\n"
/*491*/		"\n"
/*492*/		"double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }\n"
/*493*/		"\n"
/*494*/		"double func(int id, double x) {\n"
/*495*/		"	switch (id) {\n"
/*496*/		"		case 0:\n"
/*497*/		"			return sigmoid(x);\n"
/*498*/		"		case 1:\n"
/*499*/		"			return difsigmoid(x);\n"
/*500*/		"		case 2:\n"
/*501*/		"			return tanghG(x);\n"
/*502*/		"		case 3:\n"
/*503*/		"			return diftanhG(x);\n"
/*504*/		"		case 4:\n"
/*505*/		"			return relu(x);\n"
/*506*/		"		case 5:\n"
/*507*/		"			return difrelu(x);\n"
/*508*/		"		default:\n"
/*509*/		"			return 0;\n"
/*510*/		"	}\n"
/*511*/		"}\n"
/*512*/		"\n"
/*513*/		"kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,\n"
/*514*/		"			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {\n"
/*515*/		"	int m = get_global_id(0) + k0;\n"
/*516*/		"	double valorEntrada = 0;\n"
/*517*/		"	int n;\n"
/*518*/		"	for (n = 0; n < pesosy; n++) {\n"
/*519*/		"		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];\n"
/*520*/		"	}\n"
/*521*/		"	z[m] = valorEntrada;\n"
/*522*/		"	saida[m] = func(funcaoativacao, valorEntrada);\n"
/*523*/		"}\n"
/*524*/		"\n"
/*525*/		"kV fullfixweight(Vector a,\n"
/*526*/		"			  Vector pesos,\n"
/*527*/		"			  Vector dw,\n"
/*528*/		"			  Vector dz,\n"
/*529*/		"			  double hitlearn,\n"
/*530*/		"			  double decaimentoDePeso,\n"
/*531*/		"			  double momento,\n"
/*532*/		"			  int pesosy,\n"
/*533*/		"			  int k0) {\n"
/*534*/		"	int k = get_global_id(0) + k0;\n"
/*535*/		"	int m, n;\n"
/*536*/		"	m = k / pesosy;\n"
/*537*/		"	n = k % pesosy;\n"
/*538*/		"	dw[k] = dz[m] * a[n] + dw[k] * momento;\n"
/*539*/		"	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);\n"
/*540*/		"}\n"
/*541*/		"\n"
/*542*/		"kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {\n"
/*543*/		"	int m = get_global_id(0) + k0;\n"
/*544*/		"	dz[m] = ds[m] * func(dfa, z[m]);\n"
/*545*/		"}\n"
/*546*/		"\n"
/*547*/		"kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,\n"
/*548*/		"				  int k0) {\n"
/*549*/		"	int m = get_global_id(0) + k0;\n"
/*550*/		"	double soma = 0;\n"
/*551*/		"	for (int n = 0; n < pesosx; ++n) {\n"
/*552*/		"		soma += dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];\n"
/*553*/		"	}\n"
/*554*/		"	da[m] = soma;\n"
/*555*/		"}\n"
/*556*/		"\n"
/*557*/		"//padding.h\n"
/*558*/		"kV paddingfeed(Vector in,Vector out,\n"
/*559*/		"			   int txi,int tyi,\n"
/*560*/		"			   int txo,int tyo,\n"
/*561*/		"			   int t, int l ,\n"
/*562*/		"			   int k0){\n"
/*563*/		"	int k = get_global_id(0) + k0;\n"
/*564*/		"	int x, y, z;\n"
/*565*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*566*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*567*/		"	out[s] = in[k];\n"
/*568*/		"}\n"
/*569*/		"kV paddingBack(Vector gradNext,Vector gradin,\n"
/*570*/		"			   int txi,int tyi,\n"
/*571*/		"			   int txo,int tyo,\n"
/*572*/		"			   int t, int l , int k0){\n"
/*573*/		"	int k = get_global_id(0) + k0;\n"
/*574*/		"	int x, y, z;\n"
/*575*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*576*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*577*/		"	gradin[k] = gradNext[s];\n"
/*578*/		"}\n"
/*579*/		"//pool.h\n"
/*580*/		"kV poolativa(Vector entrada, Vector saida,\n"
/*581*/		"			 int passox,int passoy,\n"
/*582*/		"			 int filtrox,int filtroy,\n"
/*583*/		"			 int saidatx, int saidaty,\n"
/*584*/		"			 int entradatx, int entradaty, int k0) {\n"
/*585*/		"	int k = get_global_id(0) + k0;\n"
/*586*/		"	int x, y, z;\n"
/*587*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*588*/		"\n"
/*589*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*590*/		"	double mval, v;\n"
/*591*/		"	mval = -DBL_MAX;\n"
/*592*/		"	for (int i = 0; i < filtrox; ++i) {\n"
/*593*/		"		for (int j = 0; j < filtroy; ++j) {\n"
/*594*/		"			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*595*/		"			if (v > mval)\n"
/*596*/		"				mval = v;\n"
/*597*/		"		}\n"
/*598*/		"	}\n"
/*599*/		"	saida[k] = mval;\n"
/*600*/		"}\n"
/*601*/		"\n"
/*602*/		"\n"
/*603*/		"kV poolCalcGrads(Vector entrada, Vector gradEntrada,\n"
/*604*/		"				 Vector gradNext, Vector saida,\n"
/*605*/		"				 int fx, int fy, int px, int py,\n"
/*606*/		"				 int entradatx, int entradaty,\n"
/*607*/		"				 int saidatx, int saidaty,\n"
/*608*/		"				 int k0) {\n"
/*609*/		"	int k = get_global_id(0) + k0;\n"
/*610*/		"	int x, y, z;\n"
/*611*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*612*/		"	Range range_filtro;\n"
/*613*/		"	if (x + fx > entradatx) {\n"
/*614*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*615*/		"	}\n"
/*616*/		"	range_filtro.max.x = fx - 1;\n"
/*617*/		"	if (x - fx + 1 < 0) {\n"
/*618*/		"		range_filtro.max.x = x;\n"
/*619*/		"	}\n"
/*620*/		"	range_filtro.min.y = 0;\n"
/*621*/		"	if (y + fy > entradaty) {\n"
/*622*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*623*/		"	}\n"
/*624*/		"	range_filtro.max.y = fy - 1;\n"
/*625*/		"	if (y - fy + 1 < 0) {\n"
/*626*/		"		range_filtro.max.y = y;\n"
/*627*/		"	}\n"
/*628*/		"	int i, j;//saida\n"
/*629*/		"	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =0;\n"
/*630*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*631*/		"		i = (x - m) / px;\n"
/*632*/		"		if (i * px + m != x)continue;\n"
/*633*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*634*/		"			j = (y - n) / py;\n"
/*635*/		"			if (j * py + n != y)continue;\n"
/*636*/		"			if (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==\n"
/*637*/		"				saida[TensorMap(i, j, z, saidatx, saidaty)]) {\n"
/*638*/		"				gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =\n"
/*639*/		"						gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*640*/		"				return;\n"
/*641*/		"			}\n"
/*642*/		"		}\n"
/*643*/		"	}\n"
/*644*/		"\n"
/*645*/		"}\n"
/*646*/		"\n"
/*647*/		"\n"
/*648*/		"//poolav.h\n"
/*649*/		"kV PoolAvativa(Vector entrada, Vector saida,\n"
/*650*/		"			   int passox,int passoy,\n"
/*651*/		"			   int fx,int fy,\n"
/*652*/		"			   int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*653*/		"	int k = get_global_id(0) + k0;\n"
/*654*/		"	int x, y, z;\n"
/*655*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*656*/		"\n"
/*657*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*658*/		"	double soma = 0, v;\n"
/*659*/		"\n"
/*660*/		"	for (int i = 0; i < fx; ++i) {\n"
/*661*/		"		for (int j = 0; j < fy; ++j) {\n"
/*662*/		"			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*663*/		"		}\n"
/*664*/		"	}\n"
/*665*/		"	saida[k] = soma / (fx * fy);\n"
/*666*/		"}\n"
/*667*/		"\n"
/*668*/		"\n"
/*669*/		"kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*670*/		"                   int px, int py,  int fx, int fy,\n"
/*671*/		"				   int entradatx, int entradaty, int saidatx, int saidaty,\n"
/*672*/		"				   int k0) {\n"
/*673*/		"	int k = get_global_id(0) + k0;\n"
/*674*/		"	int x, y, z;\n"
/*675*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*676*/		"	Range range_filtro;\n"
/*677*/		"	range_filtro.min.x = 0;\n"
/*678*/		"	if (x + fx > entradatx) {\n"
/*679*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*680*/		"	}\n"
/*681*/		"	range_filtro.max.x = fx - 1;\n"
/*682*/		"	if (x - fx + 1 < 0) {\n"
/*683*/		"		range_filtro.max.x = x;\n"
/*684*/		"	}\n"
/*685*/		"	range_filtro.min.y = 0;\n"
/*686*/		"	if (y + fy > entradaty) {\n"
/*687*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*688*/		"	}\n"
/*689*/		"	range_filtro.max.y = fy - 1;\n"
/*690*/		"	if (y - fy + 1 < 0) {\n"
/*691*/		"		range_filtro.max.y = y;\n"
/*692*/		"	}\n"
/*693*/		"	int i, j;//saida\n"
/*694*/		"	double soma = 0;\n"
/*695*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*696*/		"		i = (x - m) / px;\n"
/*697*/		"		if (i * px + m != x)continue;\n"
/*698*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*699*/		"			j = (y - n) / py;\n"
/*700*/		"			if (j * py + n != y)continue;\n"
/*701*/		"			soma += gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*702*/		"		}\n"
/*703*/		"	}\n"
/*704*/		"	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);\n"
/*705*/		"\n"
/*706*/		"}\n"
/*707*/		"\n"
/*708*/		"\n"
/*709*/		"//relu.h\n"
/*710*/		"kV reluativa(Vector entrada, Vector saida, int k0) {\n"
/*711*/		"	int k = get_global_id(0) + k0;\n"
/*712*/		"	double v = entrada[k];\n"
/*713*/		"	if (v < 0)\n"
/*714*/		"		v = 0;\n"
/*715*/		"	saida[k] = v;\n"
/*716*/		"}\n"
/*717*/		"\n"
/*718*/		"kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*719*/		"	int k = get_global_id(0) + k0;\n"
/*720*/		"	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];\n"
/*721*/		"}\n"
/*722*/		"\n"
/*723*/		"//softmax.h\n"
/*724*/		"kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,\n"
/*725*/		"                 int entradaty,\n"
/*726*/		"                 int k0) {\n"
/*727*/		"	int k = get_global_id(0) + k0;\n"
/*728*/		"	int x, y, z;\n"
/*729*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*730*/		"	exponent[k] = exp(entrada[k]);\n"
/*731*/		"	soma[z] += exponent[k];\n"
/*732*/		"}\n"
/*733*/		"\n"
/*734*/		"kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,\n"
/*735*/		"                 int saidatx, int saidaty, int k0) {\n"
/*736*/		"	int k = get_global_id(0) + k0;\n"
/*737*/		"	int x, y, z;\n"
/*738*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*739*/		"	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];\n"
/*740*/		"}\n"
/*741*/		"\n"
/*742*/		"kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*743*/		"	int k = get_global_id(0) + k0;\n"
/*744*/		"	double xi = entrada[k];\n"
/*745*/		"	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];\n"
/*746*/		"}\n"
/*747*/		"\n"
/*748*/		"\n"
/*749*/		"#endif //GAB_KERNELS_OPENCL_H\n"
;
const char *getInternalDefaultKernel() {
	return __default_kernel__;
}