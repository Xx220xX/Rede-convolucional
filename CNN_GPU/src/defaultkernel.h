#ifndef KERNELS_H
#define KERNELS_H
const char default_kernel[] = 
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
/*172*/		"\n"
/*173*/		"kV\n"
/*174*/		"normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,\n"
/*175*/		"				int k0) {\n"
/*176*/		"	int k = get_global_id(0) + k0;\n"
/*177*/		"	saida[k] = (input[k] + somador) * multiplicador - subtrator;\n"
/*178*/		"}\n"
/*179*/		"\n"
/*180*/		"\n"
/*181*/		"kV subKernel(Vector grad, Vector saida, Vector target, int k0) {\n"
/*182*/		"	int k = get_global_id(0) + k0;\n"
/*183*/		"	grad[k] = saida[k] - target[k];\n"
/*184*/		"}\n"
/*185*/		"\n"
/*186*/		"kV divKernel(Vector v, double value, int k0) {\n"
/*187*/		"	int k = get_global_id(0) + k0;\n"
/*188*/		"	v[k] = v[k] / value;\n"
/*189*/		"}\n"
/*190*/		"\n"
/*191*/		"kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {\n"
/*192*/		"	int k = get_global_id(0) + k0;\n"
/*193*/		"	v[k] = ((double) src[k]) / value;\n"
/*194*/		"}\n"
/*195*/		"\n"
/*196*/		"kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {\n"
/*197*/		"	int k = get_global_id(0) + k0;\n"
/*198*/		"	for (int j = 0; j < noptiobs; j++) {\n"
/*199*/		"		v[k * noptiobs + j] = (double) (j == ints[k]);\n"
/*200*/		"	}\n"
/*201*/		"}\n"
/*202*/		"\n"
/*203*/		"\n"
/*204*/		"\n"
/*205*/		"\n"
/*206*/		"//conv.h\n"
/*207*/		"kV convSum(Vector filtro, Vector entrada, Vector saida,\n"
/*208*/		"           int passox, int passoy,\n"
/*209*/		"           int saidatx, int saidaty,\n"
/*210*/		"           int entradatx, int entradaty,\n"
/*211*/		"           int fx, int fy, int fz, int k0) {\n"
/*212*/		"	int k = get_global_id(0) + k0;\n"
/*213*/		"	int x, y, filtrok;\n"
/*214*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*215*/		"	double sum = 0, f = 0, v = 0;\n"
/*216*/		"	int lf = 0, le = 0;\n"
/*217*/		"	for (int m = 0; m < fx; m++) {\n"
/*218*/		"		for (int n = 0; n < fy; n++) {\n"
/*219*/		"			for (int z = 0; z < fz; z++) {\n"
/*220*/		"				lf = TensorMap4D(m, n, z, filtrok, fx, fy, fz);\n"
/*221*/		"				le = TensorMap(x * passox + m, y * passoy + n, z, entradatx, entradaty);\n"
/*222*/		"				f = filtro[lf];\n"
/*223*/		"				v = entrada[le];\n"
/*224*/		"				sum += f * v;\n"
/*225*/		"			}\n"
/*226*/		"		}\n"
/*227*/		"	}\n"
/*228*/		"	saida[k] = sum;\n"
/*229*/		"}\n"
/*230*/		"\n"
/*231*/		"\n"
/*232*/		"kV convCalcGradAndFixWeight(Vector filtros, Vector ds,\n"
/*233*/		"                            Vector entrada, Vector gradFiltro,\n"
/*234*/		"                            int fx, int fy, int fz,\n"
/*235*/		"                            int entrada_tx, int entrada_ty,\n"
/*236*/		"                            int saida_tx, int saida_ty,\n"
/*237*/		"                            int passox, int passoy,\n"
/*238*/		"                            double hitLearn, double momento, double weightDecay,\n"
/*239*/		"                            int k0) {\n"
/*240*/		"	int k = get_global_id(0) + k0;\n"
/*241*/		"	int m, n, z, l;\n"
/*242*/		"	TensorRemap4D(k, m, n, z, l, fx, fy, fz)\n"
/*243*/		"	double soma = 0;\n"
/*244*/		"	int le, ls;\n"
/*245*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*246*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*247*/		"			le = TensorMap(i * passox + m, j * passoy + n, z, entrada_tx, entrada_ty);\n"
/*248*/		"			ls = TensorMap(i, j, l, saida_tx, saida_ty);\n"
/*249*/		"			soma += entrada[le]\n"
/*250*/		"			        * ds[ls];\n"
/*251*/		"		}\n"
/*252*/		"	}\n"
/*253*/		"	double dw = soma + gradFiltro[k] * momento;\n"
/*254*/		"	double w = filtros[k];\n"
/*255*/		"	filtros[k] = w - hitLearn * (dw + w * weightDecay);\n"
/*256*/		"	gradFiltro[k] = dw;\n"
/*257*/		"}\n"
/*258*/		"\n"
/*259*/		"kV convCalcGradIn(Vector filtro, Vector gradEntrada, Vector gradNext,\n"
/*260*/		"                  int fx, int fy, int fz,\n"
/*261*/		"                  int passox, int passoy,\n"
/*262*/		"                  int entradatx, int entradaty,\n"
/*263*/		"                  int saidatx, int saidaty, int saidatz,\n"
/*264*/		"                  int k0) {\n"
/*265*/		"	int k = get_global_id(0) + k0;\n"
/*266*/		"	int x, y, z;\n"
/*267*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*268*/		"\n"
/*269*/		"	Range range_filtro;\n"
/*270*/		"	range_filtro.min.x = 0;\n"
/*271*/		"	if (x + fx > entradatx) {\n"
/*272*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*273*/		"	}\n"
/*274*/		"	range_filtro.max.x = fx - 1;\n"
/*275*/		"	if (x - fx + 1 < 0) {\n"
/*276*/		"		range_filtro.max.x = x;\n"
/*277*/		"	}\n"
/*278*/		"	range_filtro.min.y = 0;\n"
/*279*/		"	if (y + fy > entradaty) {\n"
/*280*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*281*/		"	}\n"
/*282*/		"	range_filtro.max.y = fy - 1;\n"
/*283*/		"	if (y - fy + 1 < 0) {\n"
/*284*/		"		range_filtro.max.y = y;\n"
/*285*/		"	}\n"
/*286*/		"	double somaErro = 0, pesoAplicado = 0;\n"
/*287*/		"	int i, j;\n"
/*288*/		"	int lf, ls;\n"
/*289*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*290*/		"		i = (x - m) / passox;\n"
/*291*/		"		if (i * passox + m != x) continue;\n"
/*292*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*293*/		"			j = (y - n) / passoy;\n"
/*294*/		"			if (j * passoy + n != y) continue;\n"
/*295*/		"			for (int w = 0; w < saidatz; w++) {\n"
/*296*/		"				lf = TensorMap4D(m, n, z, w, fx, fy, fz);\n"
/*297*/		"				ls = TensorMap(i, j, w, saidatx, saidaty);\n"
/*298*/		"				pesoAplicado = filtro[lf];\n"
/*299*/		"				somaErro += pesoAplicado * gradNext[ls];\n"
/*300*/		"			}\n"
/*301*/		"		}\n"
/*302*/		"	}\n"
/*303*/		"	gradEntrada[k] = somaErro;\n"
/*304*/		"}\n"
/*305*/		"\n"
/*306*/		"\n"
/*307*/		"//convNc.h\n"
/*308*/		"//#include\"utils.h\"\n"
/*309*/		"kV convncSum(Vector filtro, Vector entrada, Vector saida,\n"
/*310*/		"             int passox, int passoy, int largx,\n"
/*311*/		"             int largy, int saidatx, int saidaty,\n"
/*312*/		"             int entradatx, int entradaty,int fx, int fy,\n"
/*313*/		"             int entradatz, int k0) {\n"
/*314*/		"	int k = get_global_id(0) + k0;\n"
/*315*/		"	int x, y, filtrok;\n"
/*316*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*317*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*318*/		"	double sum = 0, f, v;\n"
/*319*/		"	for (int i = 0; i < fx; i++)\n"
/*320*/		"		for (int j = 0; j < fy; j++)\n"
/*321*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*322*/		"				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];\n"
/*323*/		"				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];\n"
/*324*/		"\n"
/*325*/		"				sum += f * v;\n"
/*326*/		"			}\n"
/*327*/		"	saida[k] = sum;\n"
/*328*/		"}\n"
/*329*/		"\n"
/*330*/		"kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,\n"
/*331*/		"				   double hitlearn,\n"
/*332*/		"                   double momento, double weightDecay, int k0) {\n"
/*333*/		"	int k = get_global_id(0) + k0;\n"
/*334*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*335*/		"	double w = filtro[k];\n"
/*336*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*337*/		"	gradOld[k] = m;\n"
/*338*/		"}\n"
/*339*/		"\n"
/*340*/		"kV convncCalcFiltro(Vector ds,\n"
/*341*/		"                    Vector entrada,\n"
/*342*/		"                    Vector gradFiltro,\n"
/*343*/		"                    int gradFiltro_tx,\n"
/*344*/		"                    int gradFiltro_ty,\n"
/*345*/		"                    int gradFiltro_tz,\n"
/*346*/		"\n"
/*347*/		"                    int entrada_tx,\n"
/*348*/		"                    int entrada_ty,\n"
/*349*/		"\n"
/*350*/		"                    int saida_tx,\n"
/*351*/		"                    int saida_ty,\n"
/*352*/		"\n"
/*353*/		"                    int passox,\n"
/*354*/		"                    int passoy,\n"
/*355*/		"\n"
/*356*/		"                    int largx,\n"
/*357*/		"                    int largy,\n"
/*358*/		"                    int k0) {\n"
/*359*/		"	int k = get_global_id(0) + k0;\n"
/*360*/		"	int m, n, z, l;\n"
/*361*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*362*/		"	double soma = 0,aux;\n"
/*363*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*364*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*365*/		"			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]\n"
/*366*/		"			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];\n"
/*367*/		"			//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*368*/		"			soma += aux;\n"
/*369*/		"		}\n"
/*370*/		"	}\n"
/*371*/		"	gradFiltro[k] = soma;\n"
/*372*/		"}\n"
/*373*/		"\n"
/*374*/		"/**\n"
/*375*/		" * equacao a ser implementada\n"
/*376*/		" * x = s*p + m*w\n"
/*377*/		" * onde:\n"
/*378*/		" * 	x é da entrada \n"
/*379*/		" * 	s é da saida\n"
/*380*/		" * 	m é do filtro\n"
/*381*/		" * 	s = (x - m*w)/p\n"
/*382*/		" */\n"
/*383*/		"kV convncCalcGrads(Vector filtro,\n"
/*384*/		"                   Vector entrada,\n"
/*385*/		"                   Vector gradEntrada,\n"
/*386*/		"                   Vector gradNext,\n"
/*387*/		"\n"
/*388*/		"                   int passox,\n"
/*389*/		"                   int passoy,\n"
/*390*/		"                   int largx,\n"
/*391*/		"                   int largy,\n"
/*392*/		"\n"
/*393*/		"                   int entradatx,\n"
/*394*/		"                   int entradaty,\n"
/*395*/		"                   int saidatx,\n"
/*396*/		"                   int saidaty,\n"
/*397*/		"\n"
/*398*/		"                   int fx,\n"
/*399*/		"                   int fy,\n"
/*400*/		"                   int fz,\n"
/*401*/		"                   int numFilters,\n"
/*402*/		"\n"
/*403*/		"                   int k0) {\n"
/*404*/		"	int k = get_global_id(0) + k0;\n"
/*405*/		"	int x, y, z;\n"
/*406*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*407*/		"	Range range_filtro ;\n"
/*408*/		"	range_filtro.min.x = 0;\n"
/*409*/		"	if ((entradatx - x - (fx - 1) * largx) < 0) {\n"
/*410*/		"		range_filtro.min.x = -entradatx + x + fx;\n"
/*411*/		"	}\n"
/*412*/		"	range_filtro.max.x = fx - 1;\n"
/*413*/		"	if (x - (fx - 1) * largx < 0) {\n"
/*414*/		"		range_filtro.max.x = x / largx;\n"
/*415*/		"	}\n"
/*416*/		"	range_filtro.min.y = 0;\n"
/*417*/		"	if ((entradaty - y - (fy - 1) * largy) < 0) {\n"
/*418*/		"		range_filtro.min.y = -entradaty + y + fy;\n"
/*419*/		"	}\n"
/*420*/		"	range_filtro.max.y = fy - 1;\n"
/*421*/		"	if (y - (fy - 1) * largy < 0) {\n"
/*422*/		"		range_filtro.max.y = y / largy;\n"
/*423*/		"	}\n"
/*424*/		"	int sx, sy;\n"
/*425*/		"	double somaErro = 0,aux, pesoAplicado = 0;\n"
/*426*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*427*/		"		sx = (x - m * largx) / passox;\n"
/*428*/		"		if (sx * passox + m * largx != x)continue;\n"
/*429*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*430*/		"			sy = (y - n * largy) / passox;\n"
/*431*/		"			if (sy * passoy + n * largy != y)continue;\n"
/*432*/		"			for (int l = 0; l < fz; l++) {\n"
/*433*/		"				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];\n"
/*434*/		"				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];\n"
/*435*/		"				//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*436*/		"				somaErro +=aux;\n"
/*437*/		"			}\n"
/*438*/		"		}\n"
/*439*/		"	}\n"
/*440*/		"	gradEntrada[k] = somaErro;\n"
/*441*/		"}\n"
/*442*/		"\n"
/*443*/		"\n"
/*444*/		"//dropout.h\n"
/*445*/		"#define MAX_INT_DP  ((1UL << 31) - 1)\n"
/*446*/		"long randoml(unsigned long seed,unsigned long id) {\n"
/*447*/		"	seed += id;\n"
/*448*/		"	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;\n"
/*449*/		"}\n"
/*450*/		"\n"
/*451*/		"double randomD(unsigned long seed,unsigned long id) {\n"
/*452*/		"	return (double) randoml(seed, id) / (double) MAX_INT_DP;\n"
/*453*/		"}\n"
/*454*/		"\n"
/*455*/		"kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
/*456*/		"			 double pativa, int k0) {\n"
/*457*/		"	int i = get_global_id(0) + k0;\n"
/*458*/		"//	printf(\"kernel %lf %lf %g %g\\n\",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);\n"
/*459*/		"	char teste = (char) (randomD(seed, i) <= pativa);\n"
/*460*/		"	hitmap[i] = teste;\n"
/*461*/		"	saida[i] = teste * entrada[i];\n"
/*462*/		"}\n"
/*463*/		"\n"
/*464*/		"\n"
/*465*/		"kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {\n"
/*466*/		"	int i = get_global_id(0) + k0;\n"
/*467*/		"	gradentrada[i] = hitmap[i] * gradnext[i];\n"
/*468*/		"}\n"
/*469*/		"\n"
/*470*/		"//fullconnect.h\n"
/*471*/		"double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }\n"
/*472*/		"\n"
/*473*/		"double difsigmoid(double x) {\n"
/*474*/		"	double tmp = sigmoid(x);\n"
/*475*/		"	return tmp * (1.0 - tmp);\n"
/*476*/		"}\n"
/*477*/		"\n"
/*478*/		"double tanghG(double x) { return tanh(x); }\n"
/*479*/		"\n"
/*480*/		"double diftanhG(double x) {\n"
/*481*/		"	double tmp = tanh(x);\n"
/*482*/		"	return (1.0 - tmp * tmp);\n"
/*483*/		"}\n"
/*484*/		"\n"
/*485*/		"double relu(double x) { return x > 0 ? x : 0.0; }\n"
/*486*/		"\n"
/*487*/		"double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }\n"
/*488*/		"\n"
/*489*/		"double func(int id, double x) {\n"
/*490*/		"	switch (id) {\n"
/*491*/		"		case 0:\n"
/*492*/		"			return sigmoid(x);\n"
/*493*/		"		case 1:\n"
/*494*/		"			return difsigmoid(x);\n"
/*495*/		"		case 2:\n"
/*496*/		"			return tanghG(x);\n"
/*497*/		"		case 3:\n"
/*498*/		"			return diftanhG(x);\n"
/*499*/		"		case 4:\n"
/*500*/		"			return relu(x);\n"
/*501*/		"		case 5:\n"
/*502*/		"			return difrelu(x);\n"
/*503*/		"		default:\n"
/*504*/		"			return 0;\n"
/*505*/		"	}\n"
/*506*/		"}\n"
/*507*/		"\n"
/*508*/		"kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,\n"
/*509*/		"			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {\n"
/*510*/		"	int m = get_global_id(0) + k0;\n"
/*511*/		"	double valorEntrada = 0;\n"
/*512*/		"	int n;\n"
/*513*/		"	for (n = 0; n < pesosy; n++) {\n"
/*514*/		"		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];\n"
/*515*/		"	}\n"
/*516*/		"	z[m] = valorEntrada;\n"
/*517*/		"	saida[m] = func(funcaoativacao, valorEntrada);\n"
/*518*/		"}\n"
/*519*/		"\n"
/*520*/		"kV fullfixweight(Vector a,\n"
/*521*/		"			  Vector pesos,\n"
/*522*/		"			  Vector dw,\n"
/*523*/		"			  Vector dz,\n"
/*524*/		"			  double hitlearn,\n"
/*525*/		"			  double decaimentoDePeso,\n"
/*526*/		"			  double momento,\n"
/*527*/		"			  int pesosy,\n"
/*528*/		"			  int k0) {\n"
/*529*/		"	int k = get_global_id(0) + k0;\n"
/*530*/		"	int m, n;\n"
/*531*/		"	m = k / pesosy;\n"
/*532*/		"	n = k % pesosy;\n"
/*533*/		"	dw[k] = dz[m] * a[n] + dw[k] * momento;\n"
/*534*/		"	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);\n"
/*535*/		"}\n"
/*536*/		"\n"
/*537*/		"kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {\n"
/*538*/		"	int m = get_global_id(0) + k0;\n"
/*539*/		"	dz[m] = ds[m] * func(dfa, z[m]);\n"
/*540*/		"}\n"
/*541*/		"\n"
/*542*/		"kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,\n"
/*543*/		"				  int k0) {\n"
/*544*/		"	int m = get_global_id(0) + k0;\n"
/*545*/		"	double soma = 0;\n"
/*546*/		"	for (int n = 0; n < pesosx; ++n) {\n"
/*547*/		"		soma += dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];\n"
/*548*/		"	}\n"
/*549*/		"	da[m] = soma;\n"
/*550*/		"}\n"
/*551*/		"\n"
/*552*/		"//padding.h\n"
/*553*/		"kV paddingfeed(Vector in,Vector out,\n"
/*554*/		"			   int txi,int tyi,\n"
/*555*/		"			   int txo,int tyo,\n"
/*556*/		"			   int t, int l ,\n"
/*557*/		"			   int k0){\n"
/*558*/		"	int k = get_global_id(0) + k0;\n"
/*559*/		"	int x, y, z;\n"
/*560*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*561*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*562*/		"	out[s] = in[k];\n"
/*563*/		"}\n"
/*564*/		"kV paddingBack(Vector gradNext,Vector gradin,\n"
/*565*/		"			   int txi,int tyi,\n"
/*566*/		"			   int txo,int tyo,\n"
/*567*/		"			   int t, int l , int k0){\n"
/*568*/		"	int k = get_global_id(0) + k0;\n"
/*569*/		"	int x, y, z;\n"
/*570*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*571*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*572*/		"	gradin[k] = gradNext[s];\n"
/*573*/		"}\n"
/*574*/		"//pool.h\n"
/*575*/		"kV poolativa(Vector entrada, Vector saida,\n"
/*576*/		"			 int passox,int passoy,\n"
/*577*/		"			 int filtrox,int filtroy,\n"
/*578*/		"			 int saidatx, int saidaty,\n"
/*579*/		"			 int entradatx, int entradaty, int k0) {\n"
/*580*/		"	int k = get_global_id(0) + k0;\n"
/*581*/		"	int x, y, z;\n"
/*582*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*583*/		"\n"
/*584*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*585*/		"	double mval, v;\n"
/*586*/		"	mval = -DBL_MAX;\n"
/*587*/		"	for (int i = 0; i < filtrox; ++i) {\n"
/*588*/		"		for (int j = 0; j < filtroy; ++j) {\n"
/*589*/		"			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*590*/		"			if (v > mval)\n"
/*591*/		"				mval = v;\n"
/*592*/		"		}\n"
/*593*/		"	}\n"
/*594*/		"	saida[k] = mval;\n"
/*595*/		"}\n"
/*596*/		"\n"
/*597*/		"\n"
/*598*/		"kV poolCalcGrads(Vector entrada, Vector gradEntrada,\n"
/*599*/		"				 Vector gradNext, Vector saida,\n"
/*600*/		"				 int fx, int fy, int px, int py,\n"
/*601*/		"				 int entradatx, int entradaty,\n"
/*602*/		"				 int saidatx, int saidaty,\n"
/*603*/		"				 int k0) {\n"
/*604*/		"	int k = get_global_id(0) + k0;\n"
/*605*/		"	int x, y, z;\n"
/*606*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*607*/		"	Range range_filtro;\n"
/*608*/		"	if (x + fx > entradatx) {\n"
/*609*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*610*/		"	}\n"
/*611*/		"	range_filtro.max.x = fx - 1;\n"
/*612*/		"	if (x - fx + 1 < 0) {\n"
/*613*/		"		range_filtro.max.x = x;\n"
/*614*/		"	}\n"
/*615*/		"	range_filtro.min.y = 0;\n"
/*616*/		"	if (y + fy > entradaty) {\n"
/*617*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*618*/		"	}\n"
/*619*/		"	range_filtro.max.y = fy - 1;\n"
/*620*/		"	if (y - fy + 1 < 0) {\n"
/*621*/		"		range_filtro.max.y = y;\n"
/*622*/		"	}\n"
/*623*/		"	int i, j;//saida\n"
/*624*/		"	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =0;\n"
/*625*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*626*/		"		i = (x - m) / px;\n"
/*627*/		"		if (i * px + m != x)continue;\n"
/*628*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*629*/		"			j = (y - n) / py;\n"
/*630*/		"			if (j * py + n != y)continue;\n"
/*631*/		"			if (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==\n"
/*632*/		"				saida[TensorMap(i, j, z, saidatx, saidaty)]) {\n"
/*633*/		"				gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] =\n"
/*634*/		"						gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*635*/		"				return;\n"
/*636*/		"			}\n"
/*637*/		"		}\n"
/*638*/		"	}\n"
/*639*/		"\n"
/*640*/		"}\n"
/*641*/		"\n"
/*642*/		"\n"
/*643*/		"//poolav.h\n"
/*644*/		"kV PoolAvativa(Vector entrada, Vector saida,\n"
/*645*/		"			   int passox,int passoy,\n"
/*646*/		"			   int fx,int fy,\n"
/*647*/		"			   int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*648*/		"	int k = get_global_id(0) + k0;\n"
/*649*/		"	int x, y, z;\n"
/*650*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*651*/		"\n"
/*652*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*653*/		"	double soma = 0, v;\n"
/*654*/		"\n"
/*655*/		"	for (int i = 0; i < fx; ++i) {\n"
/*656*/		"		for (int j = 0; j < fy; ++j) {\n"
/*657*/		"			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*658*/		"		}\n"
/*659*/		"	}\n"
/*660*/		"	saida[k] = soma / (fx * fy);\n"
/*661*/		"}\n"
/*662*/		"\n"
/*663*/		"\n"
/*664*/		"kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*665*/		"                   int px, int py,  int fx, int fy,\n"
/*666*/		"				   int entradatx, int entradaty, int saidatx, int saidaty,\n"
/*667*/		"				   int k0) {\n"
/*668*/		"	int k = get_global_id(0) + k0;\n"
/*669*/		"	int x, y, z;\n"
/*670*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*671*/		"	Range range_filtro;\n"
/*672*/		"	range_filtro.min.x = 0;\n"
/*673*/		"	if (x + fx > entradatx) {\n"
/*674*/		"		range_filtro.min.x = x + fx - entradatx;\n"
/*675*/		"	}\n"
/*676*/		"	range_filtro.max.x = fx - 1;\n"
/*677*/		"	if (x - fx + 1 < 0) {\n"
/*678*/		"		range_filtro.max.x = x;\n"
/*679*/		"	}\n"
/*680*/		"	range_filtro.min.y = 0;\n"
/*681*/		"	if (y + fy > entradaty) {\n"
/*682*/		"		range_filtro.min.y = y + fy - entradaty;\n"
/*683*/		"	}\n"
/*684*/		"	range_filtro.max.y = fy - 1;\n"
/*685*/		"	if (y - fy + 1 < 0) {\n"
/*686*/		"		range_filtro.max.y = y;\n"
/*687*/		"	}\n"
/*688*/		"	int i, j;//saida\n"
/*689*/		"	double soma = 0;\n"
/*690*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*691*/		"		i = (x - m) / px;\n"
/*692*/		"		if (i * px + m != x)continue;\n"
/*693*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*694*/		"			j = (y - n) / py;\n"
/*695*/		"			if (j * py + n != y)continue;\n"
/*696*/		"			soma += gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*697*/		"		}\n"
/*698*/		"	}\n"
/*699*/		"	gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = soma / (fx * fy);\n"
/*700*/		"\n"
/*701*/		"}\n"
/*702*/		"\n"
/*703*/		"\n"
/*704*/		"//relu.h\n"
/*705*/		"kV reluativa(Vector entrada, Vector saida, int k0) {\n"
/*706*/		"	int k = get_global_id(0) + k0;\n"
/*707*/		"	double v = entrada[k];\n"
/*708*/		"	if (v < 0)\n"
/*709*/		"		v = 0;\n"
/*710*/		"	saida[k] = v;\n"
/*711*/		"}\n"
/*712*/		"\n"
/*713*/		"kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*714*/		"	int k = get_global_id(0) + k0;\n"
/*715*/		"	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];\n"
/*716*/		"}\n"
/*717*/		"\n"
/*718*/		"//softmax.h\n"
/*719*/		"kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,\n"
/*720*/		"                 int entradaty,\n"
/*721*/		"                 int k0) {\n"
/*722*/		"	int k = get_global_id(0) + k0;\n"
/*723*/		"	int x, y, z;\n"
/*724*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*725*/		"	exponent[k] = exp(entrada[k]);\n"
/*726*/		"	soma[z] += exponent[k];\n"
/*727*/		"}\n"
/*728*/		"\n"
/*729*/		"kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,\n"
/*730*/		"                 int saidatx, int saidaty, int k0) {\n"
/*731*/		"	int k = get_global_id(0) + k0;\n"
/*732*/		"	int x, y, z;\n"
/*733*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*734*/		"	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];\n"
/*735*/		"}\n"
/*736*/		"\n"
/*737*/		"kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*738*/		"	int k = get_global_id(0) + k0;\n"
/*739*/		"	double xi = entrada[k];\n"
/*740*/		"	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];\n"
/*741*/		"}\n"
/*742*/		"\n"
/*743*/		"\n"
/*744*/		"#endif //GAB_KERNELS_OPENCL_H\n"
;
#endif // KERNELS_H
