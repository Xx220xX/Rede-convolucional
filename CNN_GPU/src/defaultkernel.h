#ifndef KERNELS_H
#define KERNELS_H
const char default_kernel[] = 
/*1*/		"//utils.h\n"
/*2*/		"// Created by Xx220xX on 10/05/2020.\n"
/*3*/		"\n"
/*4*/		"#define Vector __global double *\n"
/*5*/		"\n"
/*6*/		"#define kV __kernel void\n"
/*7*/		"\n"
/*8*/		"#define TensorMap(x, y, z, tx, ty)((z)*(ty*tx)+(x)*ty+(y))\n"
/*9*/		"\n"
/*10*/		"#define TensorMap4D(x, y, z, l, tx, ty, tz)((l)*(ty)*(tx)*(tz)+(z)*(ty*tx)+(x)*ty+(y))\n"
/*11*/		"\n"
/*12*/		"#define TensorRemap4D(total, _x_, _y_, _z_, _l_, tx, ty, tz)\\\n"
/*13*/		"_y_ = total%ty      ;                                        \\\n"
/*14*/		"_x_ = (total - _y_)%(ty*tx)/ty ;                             \\\n"
/*15*/		"_z_ = (total- _x_*ty - _y_)%(tx*ty*tz)/(ty*tx)  ;            \\\n"
/*16*/		"_l_ = (total -_z_*tx*ty -_x_*ty - _y_)/(tx*ty*tz);\n"
/*17*/		"\n"
/*18*/		"\n"
/*19*/		"#define TensorRemap(total, _x_, _y_, _z_, tx, ty)\\\n"
/*20*/		"_y_ = total % ty;\\\n"
/*21*/		"_x_ = ((total - _y_) % (ty * tx)) / ty;\\\n"
/*22*/		"_z_ = (k - _x_ * ty - _y_) / (tx * ty);\n"
/*23*/		"\n"
/*24*/		"#define TensorRemap2D(total, x, y, ty)\\\n"
/*25*/		"y = total % ty;\\\n"
/*26*/		"x = total/ ty;\n"
/*27*/		"\n"
/*28*/		"typedef struct {\n"
/*29*/		"	int x, y, z;\n"
/*30*/		"} Ponto3d;\n"
/*31*/		"\n"
/*32*/		"typedef struct {\n"
/*33*/		"	Ponto3d min, max;\n"
/*34*/		"} Range;\n"
/*35*/		"\n"
/*36*/		"kV createImg(__global unsigned char *out, Vector v, int vx, int vy, int imi, int imy, int k0) {\n"
/*37*/		"	int k = get_global_id(0) + k0;\n"
/*38*/		"	int i, j, z;\n"
/*39*/		"	TensorRemap(k, i, j, z, vx, vy)\n"
/*40*/		"	imi = imi + i;\n"
/*41*/		"	int imj = j + z * vy + z;\n"
/*42*/		"	out[imi * imy + imj] = ((int) v[k]) & 0xff;\n"
/*43*/		"}\n"
/*44*/		"\n"
/*45*/		"kV printTensor(Vector t, int mx, int my, int mz, int ofset) {\n"
/*46*/		"	for (int z = 0; z < mz; z++) {\n"
/*47*/		"		printf(\"[Dim%d]\\n\", z);\n"
/*48*/		"		for (int x = 0; x < mx; x++) {\n"
/*49*/		"			for (int y = 0; y < my; y++) {\n"
/*50*/		"\n"
/*51*/		"				printf(\"%.4lf \\t\", t[TensorMap(x, y, z, mx, my) + ofset]);\n"
/*52*/		"			}\n"
/*53*/		"			printf(\"\\n\");\n"
/*54*/		"		}\n"
/*55*/		"	}\n"
/*56*/		"}\n"
/*57*/		"\n"
/*58*/		"kV norm(Vector v, Vector out, int len) {\n"
/*59*/		"	double s = 0,aux;\n"
/*60*/		"	for (int i = 0; i < len; ++i) {\n"
/*61*/		"		aux = v[i] * v[i];\n"
/*62*/		"		aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*63*/		"		s += aux;\n"
/*64*/		"	}\n"
/*65*/		"	out[0] = pow(s, 0.5);\n"
/*66*/		"}\n"
/*67*/		"\n"
/*68*/		"kV maxID(Vector v, Vector out, int len) {\n"
/*69*/		"	int s = 0;\n"
/*70*/		"	for (int i = 1; i < len; ++i) {\n"
/*71*/		"		if (v[s] < v[i]) {\n"
/*72*/		"			s = i;\n"
/*73*/		"		}\n"
/*74*/		"	}\n"
/*75*/		"	out[0] = (double) s;\n"
/*76*/		"}\n"
/*77*/		"\n"
/*78*/		"kV\n"
/*79*/		"normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,\n"
/*80*/		"                int k0) {\n"
/*81*/		"	int k = get_global_id(0) + k0;\n"
/*82*/		"	saida[k] = (input[k] + somador) * multiplicador - subtrator;\n"
/*83*/		"}\n"
/*84*/		"\n"
/*85*/		"kV findExtremes(Vector input, Vector output, int len) {\n"
/*86*/		"	double mn = input[0], mx = input[0];\n"
/*87*/		"	for (int i = 1; i < len; ++i) {\n"
/*88*/		"		if (input[i] > mx) mx = input[i];\n"
/*89*/		"		if (input[i] < mn) mn = input[i];\n"
/*90*/		"	}\n"
/*91*/		"	output[0] = mn;\n"
/*92*/		"	output[1] = mx;\n"
/*93*/		"}\n"
/*94*/		"\n"
/*95*/		"kV sub(Vector grad, Vector saida, Vector target, int k0) {\n"
/*96*/		"	int k = get_global_id(0) + k0;\n"
/*97*/		"\n"
/*98*/		"	grad[k] = saida[k] - target[k];\n"
/*99*/		"}\n"
/*100*/		"\n"
/*101*/		"kV div(Vector v, double value, int k0) {\n"
/*102*/		"	int k = get_global_id(0) + k0;\n"
/*103*/		"	v[k] = v[k] / value;\n"
/*104*/		"}\n"
/*105*/		"\n"
/*106*/		"kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {\n"
/*107*/		"	int k = get_global_id(0) + k0;\n"
/*108*/		"	v[k] = ((double) src[k]) / value;\n"
/*109*/		"}\n"
/*110*/		"\n"
/*111*/		"kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {\n"
/*112*/		"	int k = get_global_id(0) + k0;\n"
/*113*/		"	for (int j = 0; j < noptiobs; j++) {\n"
/*114*/		"		v[k * noptiobs + j] = (double) (j == ints[k]);\n"
/*115*/		"	}\n"
/*116*/		"}\n"
/*117*/		"\n"
/*118*/		"\n"
/*119*/		"int normaliza_range(double f, int max, int lim_min) {\n"
/*120*/		"	if (f <= 0)return 0;\n"
/*121*/		"	if (f >= max - 1)return max - 1;\n"
/*122*/		"	if (lim_min) return ceil(f);\n"
/*123*/		"	else return floor(f);\n"
/*124*/		"}\n"
/*125*/		"\n"
/*126*/		"Range mapeia_entrada_saida(int x, int y, int passo, int tamanhoFiltro, int saidatx, int saidaty, int numeroFiltros) {\n"
/*127*/		"	double a = x, b = y;\n"
/*128*/		"	Range r;\n"
/*129*/		"	r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saidatx, 1);\n"
/*130*/		"	r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saidaty, 1);\n"
/*131*/		"	r.min.z = 0;\n"
/*132*/		"\n"
/*133*/		"	r.max.x = normaliza_range(a / passo, saidatx, 0);\n"
/*134*/		"	r.max.y = normaliza_range(b / passo, saidaty, 0);\n"
/*135*/		"	r.max.z = numeroFiltros - 1;\n"
/*136*/		"	return r;\n"
/*137*/		"}\n"
/*138*/		"\n"
/*139*/		"//bathnorm.h\n"
/*140*/		"\n"
/*141*/		"// achar a media\n"
/*142*/		"kV BatchNormMedia(Vector entrada, Vector media,\n"
/*143*/		"                  int entradatx, int entradaty, int k0) {\n"
/*144*/		"	int z = get_global_id(0) + k0;\n"
/*145*/		"	int x, y;\n"
/*146*/		"	double m = 0;\n"
/*147*/		"	for (x = 0; x < entradatx; x++) {\n"
/*148*/		"		for (y = 0; y < entradaty; y++) {\n"
/*149*/		"			m += entrada[TensorMap(x, y, z, entradatx, entradaty)];\n"
/*150*/		"		}\n"
/*151*/		"	}\n"
/*152*/		"	media[z] = m / (double) (entradatx * entradaty);\n"
/*153*/		"}\n"
/*154*/		"\n"
/*155*/		"// achar a diferenca\n"
/*156*/		"kV BatchNormDiferenca(Vector entrada, Vector media,\n"
/*157*/		"                      Vector diferenca,\n"
/*158*/		"                      Vector diferencaquad,\n"
/*159*/		"                      int entradatx, int entradaty, int k0) {\n"
/*160*/		"	int x, y, z;\n"
/*161*/		"	int k = get_global_id(0) + k0;\n"
/*162*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*163*/		"	diferenca[k] = entrada[k] - media[z];\n"
/*164*/		"	diferencaquad[k] = diferenca[k] * diferenca[k];\n"
/*165*/		"}\n"
/*166*/		"\n"
/*167*/		"kV BatchNormVariance(Vector dif, Vector difQuad,\n"
/*168*/		"                     Vector sumdiferenca, Vector variancia,\n"
/*169*/		"                     double episolon, int diftx, int difty,\n"
/*170*/		"                     int k0) {\n"
/*171*/		"	int z = get_global_id(0) + k0;\n"
/*172*/		"	double sum = 0;\n"
/*173*/		"	double sumdif = 0;\n"
/*174*/		"	for (int x = 0; x < diftx; x++) {\n"
/*175*/		"		for (int y = 0; y < difty; y++) {\n"
/*176*/		"			sum += difQuad[TensorMap(x, y, z, diftx, difty)];\n"
/*177*/		"			sumdif += dif[TensorMap(x, y, z, diftx, difty)];\n"
/*178*/		"		}\n"
/*179*/		"	}\n"
/*180*/		"	sumdiferenca[z] = sumdif;\n"
/*181*/		"	variancia[z] = sqrt(sum / (difty * diftx) + episolon);\n"
/*182*/		"}\n"
/*183*/		"\n"
/*184*/		"// normaliza\n"
/*185*/		"kV BatchNormNormaliza(Vector saida,\n"
/*186*/		"                      Vector norma,\n"
/*187*/		"                      Vector diferenca,\n"
/*188*/		"                      Vector variancia,\n"
/*189*/		"                      Vector Y,\n"
/*190*/		"                      Vector B,\n"
/*191*/		"                      int diferencatx, int diferencaty, int k0) {\n"
/*192*/		"	int x, y, z;\n"
/*193*/		"	int k = get_global_id(0) + k0;\n"
/*194*/		"	TensorRemap(k, x, y, z, diferencatx, diferencaty)\n"
/*195*/		"	norma[k] = diferenca[k] / variancia[z];\n"
/*196*/		"	saida[k] = norma[k] * Y[z] + B[z];\n"
/*197*/		"}\n"
/*198*/		"\n"
/*199*/		"\n"
/*200*/		"kV BatchNormaCalcGrad1(Vector gradIn,\n"
/*201*/		"                       Vector gradNext,\n"
/*202*/		"                       Vector variancia,\n"
/*203*/		"                       Vector media,\n"
/*204*/		"                       Vector Y,\n"
/*205*/		"\n"
/*206*/		"                       Vector somaDif,\n"
/*207*/		"                       Vector entrada,\n"
/*208*/		"                       int entradatx,\n"
/*209*/		"                       int entradaty,\n"
/*210*/		"                       int k0) {\n"
/*211*/		"	int x, y, z;\n"
/*212*/		"	int k = get_global_id(0) + k0;\n"
/*213*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*214*/		"	double M = entradatx * entradaty;\n"
/*215*/		"	double dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);\n"
/*216*/		"	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);\n"
/*217*/		"\n"
/*218*/		"	double didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;\n"
/*219*/		"	didx = didx / (variancia[z] * variancia[z]);\n"
/*220*/		"	didx = didx * gradNext[k];\n"
/*221*/		"	gradIn[k] = didx * Y[z];\n"
/*222*/		"}\n"
/*223*/		"\n"
/*224*/		"kV BatchNormaCalcGrad2(Vector gradNext,\n"
/*225*/		"                       Vector norma,\n"
/*226*/		"                       Vector gradY,\n"
/*227*/		"                       Vector gradB,\n"
/*228*/		"                       int entradatx,\n"
/*229*/		"                       int entradaty,\n"
/*230*/		"                       int k0) {\n"
/*231*/		"	int z = get_global_id(0) + k0;\n"
/*232*/		"	double sumY = 0;\n"
/*233*/		"	double sumB = 0;\n"
/*234*/		"	int k;\n"
/*235*/		"	for (int x = 0; x < entradatx; ++x) {\n"
/*236*/		"		for (int y = 0; y < entradaty; ++y) {\n"
/*237*/		"			k = TensorMap(x, y, z, entradatx, entradaty);\n"
/*238*/		"			sumY += gradNext[k];\n"
/*239*/		"			sumB += gradNext[k] * norma[k];\n"
/*240*/		"		}\n"
/*241*/		"	}\n"
/*242*/		"	gradB[z] = sumB;\n"
/*243*/		"	gradY[z] = sumY;\n"
/*244*/		"}\n"
/*245*/		"\n"
/*246*/		"\n"
/*247*/		"kV batchNormCorrigePeso(Vector gradY,\n"
/*248*/		"                        Vector gradB,\n"
/*249*/		"                        Vector Y,\n"
/*250*/		"                        Vector B,\n"
/*251*/		"                        double hitlearn,\n"
/*252*/		"                        int k0) {\n"
/*253*/		"	int z = get_global_id(0) + k0;\n"
/*254*/		"	B[z] = B[z] - gradB[z] * hitlearn;\n"
/*255*/		"	Y[z] = Y[z] - gradY[z] * hitlearn;\n"
/*256*/		"}\n"
/*257*/		"//conv.h\n"
/*258*/		"//#include\"utils.h\"\n"
/*259*/		"kV convSum(Vector filtro, Vector entrada, Vector saida,\n"
/*260*/		"           int passo, int saidatx, int saidaty, int entradatx, int entradaty,\n"
/*261*/		"           int lenFilter, int entradatz, int k0) {\n"
/*262*/		"	int k = get_global_id(0) + k0;\n"
/*263*/		"	int x, y, filtrok;\n"
/*264*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*265*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*266*/		"	double sum = 0, f, v;\n"
/*267*/		"	for (int i = 0; i < lenFilter; i++)\n"
/*268*/		"		for (int j = 0; j < lenFilter; j++)\n"
/*269*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*270*/		"				f = filtro[TensorMap4D(i, j, z, filtrok, lenFilter, lenFilter, entradatz)];\n"
/*271*/		"				v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*272*/		"				sum += f * v;\n"
/*273*/		"			}\n"
/*274*/		"	saida[k] = sum;\n"
/*275*/		"}\n"
/*276*/		"\n"
/*277*/		"kV convFixWeight(Vector filtro, Vector grad, Vector gradOld, double hitlearn,\n"
/*278*/		"                 double momento, double weightDecay, int k0) {\n"
/*279*/		"	int k = get_global_id(0) + k0;\n"
/*280*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*281*/		"	double w = filtro[k];\n"
/*282*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*283*/		"	gradOld[k] = m;\n"
/*284*/		"}\n"
/*285*/		"\n"
/*286*/		"kV convCalcFiltro(     Vector ds,\n"
/*287*/		"					   Vector entrada,\n"
/*288*/		"					   Vector gradFiltro,\n"
/*289*/		"                       int gradFiltro_tx,\n"
/*290*/		"                       int gradFiltro_ty,\n"
/*291*/		"                       int gradFiltro_tz,\n"
/*292*/		"                       int entrada_tx,\n"
/*293*/		"                       int entrada_ty,\n"
/*294*/		"                       int saida_tx,\n"
/*295*/		"                       int saida_ty,\n"
/*296*/		"                       int passo,\n"
/*297*/		"                       int k0) {\n"
/*298*/		"	int k = get_global_id(0) + k0;\n"
/*299*/		"	int m, n, z, l;\n"
/*300*/		"//	printf(\"kernel %d\\n\",k);\n"
/*301*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*302*/		"	double soma = 0,aux;\n"
/*303*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*304*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*305*/		"			aux = entrada[TensorMap(i*passo+m, j*passo+n,z,entrada_tx,entrada_ty)]\n"
/*306*/		"				   *ds[TensorMap(i,j,l,saida_tx,saida_ty)];\n"
/*307*/		"			aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*308*/		"			soma +=aux;\n"
/*309*/		"		}\n"
/*310*/		"	}\n"
/*311*/		"	gradFiltro[k] = soma;\n"
/*312*/		"}\n"
/*313*/		"\n"
/*314*/		"kV convCalcGrads(Vector filtro,\n"
/*315*/		"				 Vector entrada,\n"
/*316*/		"                 Vector gradEntrada,\n"
/*317*/		"                 Vector gradNext,\n"
/*318*/		"                 int lenFilter,\n"
/*319*/		"                 int filtroz,\n"
/*320*/		"                 int passo,\n"
/*321*/		"                 int entradatx,\n"
/*322*/		"                 int entradaty,\n"
/*323*/		"                 int saidatx,\n"
/*324*/		"                 int saidaty,\n"
/*325*/		"                 int numFilters,\n"
/*326*/		"                 int k0) {\n"
/*327*/		"	int k = get_global_id(0) + k0;\n"
/*328*/		"	int x, y, z;\n"
/*329*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*330*/		"	Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);\n"
/*331*/		"	int minX, minY;\n"
/*332*/		"	double somaErro = 0, pesoAplicado = 0;\n"
/*333*/		"	double aux;\n"
/*334*/		"	for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*335*/		"		minX = i * passo;\n"
/*336*/		"		for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*337*/		"			minY = j * passo;\n"
/*338*/		"			for (int l = range.min.z; l <= range.max.z; l++) {\n"
/*339*/		"				pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)];\n"
/*340*/		"				aux = pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];\n"
/*341*/		"				aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*342*/		"				somaErro +=aux;\n"
/*343*/		"			}\n"
/*344*/		"		}\n"
/*345*/		"	}\n"
/*346*/		"	gradEntrada[k] = somaErro;\n"
/*347*/		"}\n"
/*348*/		"\n"
/*349*/		"\n"
/*350*/		"//convNc.h\n"
/*351*/		"//#include\"utils.h\"\n"
/*352*/		"kV convncSum(Vector filtro, Vector entrada, Vector saida,\n"
/*353*/		"             int passox, int passoy, int largx,\n"
/*354*/		"             int largy, int saidatx, int saidaty,\n"
/*355*/		"             int entradatx, int entradaty,int fx, int fy,\n"
/*356*/		"             int entradatz, int k0) {\n"
/*357*/		"	int k = get_global_id(0) + k0;\n"
/*358*/		"	int x, y, filtrok;\n"
/*359*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*360*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*361*/		"	double sum = 0, f, v;\n"
/*362*/		"	for (int i = 0; i < fx; i++)\n"
/*363*/		"		for (int j = 0; j < fy; j++)\n"
/*364*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*365*/		"				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];\n"
/*366*/		"				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];\n"
/*367*/		"\n"
/*368*/		"				sum += f * v;\n"
/*369*/		"			}\n"
/*370*/		"	saida[k] = sum;\n"
/*371*/		"}\n"
/*372*/		"\n"
/*373*/		"kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,\n"
/*374*/		"				   double hitlearn,\n"
/*375*/		"                   double momento, double weightDecay, int k0) {\n"
/*376*/		"	int k = get_global_id(0) + k0;\n"
/*377*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*378*/		"	double w = filtro[k];\n"
/*379*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*380*/		"	gradOld[k] = m;\n"
/*381*/		"}\n"
/*382*/		"\n"
/*383*/		"kV convncCalcFiltro(Vector ds,\n"
/*384*/		"                    Vector entrada,\n"
/*385*/		"                    Vector gradFiltro,\n"
/*386*/		"                    int gradFiltro_tx,\n"
/*387*/		"                    int gradFiltro_ty,\n"
/*388*/		"                    int gradFiltro_tz,\n"
/*389*/		"\n"
/*390*/		"                    int entrada_tx,\n"
/*391*/		"                    int entrada_ty,\n"
/*392*/		"\n"
/*393*/		"                    int saida_tx,\n"
/*394*/		"                    int saida_ty,\n"
/*395*/		"\n"
/*396*/		"                    int passox,\n"
/*397*/		"                    int passoy,\n"
/*398*/		"\n"
/*399*/		"                    int largx,\n"
/*400*/		"                    int largy,\n"
/*401*/		"                    int k0) {\n"
/*402*/		"	int k = get_global_id(0) + k0;\n"
/*403*/		"	int m, n, z, l;\n"
/*404*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*405*/		"	double soma = 0,aux;\n"
/*406*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*407*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*408*/		"			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]\n"
/*409*/		"			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];\n"
/*410*/		"			aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*411*/		"			soma += aux;\n"
/*412*/		"		}\n"
/*413*/		"	}\n"
/*414*/		"	gradFiltro[k] = soma;\n"
/*415*/		"}\n"
/*416*/		"\n"
/*417*/		"/**\n"
/*418*/		" * equacao a ser implementada\n"
/*419*/		" * x = s*p + m*w\n"
/*420*/		" * onde:\n"
/*421*/		" * 	x é da entrada \n"
/*422*/		" * 	s é da saida\n"
/*423*/		" * 	m é do filtro\n"
/*424*/		" * 	s = (x - m*w)/p\n"
/*425*/		" */\n"
/*426*/		"kV convncCalcGrads(Vector filtro,\n"
/*427*/		"                   Vector entrada,\n"
/*428*/		"                   Vector gradEntrada,\n"
/*429*/		"                   Vector gradNext,\n"
/*430*/		"\n"
/*431*/		"                   int passox,\n"
/*432*/		"                   int passoy,\n"
/*433*/		"                   int largx,\n"
/*434*/		"                   int largy,\n"
/*435*/		"\n"
/*436*/		"                   int entradatx,\n"
/*437*/		"                   int entradaty,\n"
/*438*/		"                   int saidatx,\n"
/*439*/		"                   int saidaty,\n"
/*440*/		"\n"
/*441*/		"                   int fx,\n"
/*442*/		"                   int fy,\n"
/*443*/		"                   int fz,\n"
/*444*/		"                   int numFilters,\n"
/*445*/		"\n"
/*446*/		"                   int k0) {\n"
/*447*/		"	int k = get_global_id(0) + k0;\n"
/*448*/		"	int x, y, z;\n"
/*449*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*450*/		"	Range range_filtro ;\n"
/*451*/		"	range_filtro.min.x = 0;\n"
/*452*/		"	if ((entradatx - x - (fx - 1) * largx) < 0) {\n"
/*453*/		"		range_filtro.min.x = -entradatx + x + fx;\n"
/*454*/		"	}\n"
/*455*/		"	range_filtro.max.x = fx - 1;\n"
/*456*/		"	if (x - (fx - 1) * largx < 0) {\n"
/*457*/		"		range_filtro.max.x = x / largx;\n"
/*458*/		"	}\n"
/*459*/		"	range_filtro.min.y = 0;\n"
/*460*/		"	if ((entradaty - y - (fy - 1) * largy) < 0) {\n"
/*461*/		"		range_filtro.min.y = -entradaty + y + fy;\n"
/*462*/		"	}\n"
/*463*/		"	range_filtro.max.y = fy - 1;\n"
/*464*/		"	if (y - (fy - 1) * largy < 0) {\n"
/*465*/		"		range_filtro.max.y = y / largy;\n"
/*466*/		"	}\n"
/*467*/		"	int sx, sy;\n"
/*468*/		"	double somaErro = 0,aux, pesoAplicado = 0;\n"
/*469*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*470*/		"		sx = (x - m * largx) / passox;\n"
/*471*/		"		if (sx * passox + m * largx != x)continue;\n"
/*472*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*473*/		"			sy = (y - n * largy) / passox;\n"
/*474*/		"			if (sy * passoy + n * largy != y)continue;\n"
/*475*/		"			for (int l = 0; l < fz; l++) {\n"
/*476*/		"				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];\n"
/*477*/		"				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];\n"
/*478*/		"				aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*479*/		"				somaErro +=aux;\n"
/*480*/		"			}\n"
/*481*/		"		}\n"
/*482*/		"	}\n"
/*483*/		"	gradEntrada[k] = somaErro;\n"
/*484*/		"}\n"
/*485*/		"\n"
/*486*/		"\n"
/*487*/		"//dropout.h\n"
/*488*/		"#define MAX_INT_DP  ((1UL << 31) - 1)\n"
/*489*/		"long randoml(unsigned long seed,unsigned long id) {\n"
/*490*/		"	seed += id;\n"
/*491*/		"	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;\n"
/*492*/		"}\n"
/*493*/		"\n"
/*494*/		"double randomD(unsigned long seed,unsigned long id) {\n"
/*495*/		"	return (double) randoml(seed, id) / (double) MAX_INT_DP;\n"
/*496*/		"}\n"
/*497*/		"\n"
/*498*/		"kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
/*499*/		"			 double pativa, int k0) {\n"
/*500*/		"	int i = get_global_id(0) + k0;\n"
/*501*/		"//	printf(\"kernel %lf %lf %g %g\\n\",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);\n"
/*502*/		"	char teste = (char) (randomD(seed, i) <= pativa);\n"
/*503*/		"	hitmap[i] = teste;\n"
/*504*/		"	saida[i] = teste * entrada[i];\n"
/*505*/		"}\n"
/*506*/		"\n"
/*507*/		"\n"
/*508*/		"kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {\n"
/*509*/		"	int i = get_global_id(0) + k0;\n"
/*510*/		"	gradentrada[i] = hitmap[i] * gradnext[i];\n"
/*511*/		"}\n"
/*512*/		"\n"
/*513*/		"//fullconnect.h\n"
/*514*/		"double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }\n"
/*515*/		"\n"
/*516*/		"double difsigmoid(double x) {\n"
/*517*/		"	double tmp = sigmoid(x);\n"
/*518*/		"	return tmp * (1.0 - tmp);\n"
/*519*/		"}\n"
/*520*/		"\n"
/*521*/		"double tanghG(double x) { return tanh(x); }\n"
/*522*/		"\n"
/*523*/		"double diftanhG(double x) {\n"
/*524*/		"	double tmp = tanh(x);\n"
/*525*/		"	return (1.0 - tmp * tmp);\n"
/*526*/		"}\n"
/*527*/		"\n"
/*528*/		"double relu(double x) { return x > 0 ? x : 0.0; }\n"
/*529*/		"\n"
/*530*/		"double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }\n"
/*531*/		"\n"
/*532*/		"double func(int id, double x) {\n"
/*533*/		"	switch (id) {\n"
/*534*/		"		case 0:\n"
/*535*/		"			return sigmoid(x);\n"
/*536*/		"		case 1:\n"
/*537*/		"			return difsigmoid(x);\n"
/*538*/		"		case 2:\n"
/*539*/		"			return tanghG(x);\n"
/*540*/		"		case 3:\n"
/*541*/		"			return diftanhG(x);\n"
/*542*/		"		case 4:\n"
/*543*/		"			return relu(x);\n"
/*544*/		"		case 5:\n"
/*545*/		"			return difrelu(x);\n"
/*546*/		"		default:\n"
/*547*/		"			return 0;\n"
/*548*/		"	}\n"
/*549*/		"}\n"
/*550*/		"\n"
/*551*/		"kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,\n"
/*552*/		"            int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {\n"
/*553*/		"	int m = get_global_id(0) + k0;\n"
/*554*/		"	double valorEntrada = 0;\n"
/*555*/		"	int n;\n"
/*556*/		"	for (n = 0; n < pesosy; n++) {\n"
/*557*/		"		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];\n"
/*558*/		"	}\n"
/*559*/		"	z[m] = valorEntrada;\n"
/*560*/		"	saida[m] = func(funcaoativacao, valorEntrada);\n"
/*561*/		"}\n"
/*562*/		"\n"
/*563*/		"kV\n"
/*564*/		"fullfixweight(Vector a,\n"
/*565*/		"              Vector pesos,\n"
/*566*/		"              Vector dz,\n"
/*567*/		"              Vector dz_old,\n"
/*568*/		"              double hitlearn,\n"
/*569*/		"              double decaimentoDePeso,\n"
/*570*/		"              double momento,\n"
/*571*/		"              int inx,\n"
/*572*/		"              int iny,\n"
/*573*/		"              int inz,\n"
/*574*/		"              int pesosx,\n"
/*575*/		"              int pesosy,\n"
/*576*/		"              int k0) {\n"
/*577*/		"	int n = get_global_id(0) + k0;\n"
/*578*/		"	int m;\n"
/*579*/		"	double w;\n"
/*580*/		"	double tmp = dz[n] + dz_old[n] * momento;\n"
/*581*/		"	dz_old[n] = tmp;\n"
/*582*/		"	int k;\n"
/*583*/		"	for (m = inx * iny * inz - 1; m >= 0; m--) {\n"
/*584*/		"		k = TensorMap(n, m, 0, pesosx, pesosy);\n"
/*585*/		"		w = pesos[k];\n"
/*586*/		"		w -= hitlearn * (tmp * a[m] + w * decaimentoDePeso);\n"
/*587*/		"		pesos[k] = w;\n"
/*588*/		"	}\n"
/*589*/		"}\n"
/*590*/		"\n"
/*591*/		"kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {\n"
/*592*/		"	int m = get_global_id(0) + k0;\n"
/*593*/		"	double aux = ds[m] * func(dfa, z[m]);\n"
/*594*/		"	aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*595*/		"	dz[m] = aux;\n"
/*596*/		"}\n"
/*597*/		"\n"
/*598*/		"kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,\n"
/*599*/		"                  int k0) {\n"
/*600*/		"	int m = get_global_id(0) + k0;\n"
/*601*/		"	double soma = 0,aux;\n"
/*602*/		"	for (int n = 0; n < pesosx; ++n) {\n"
/*603*/		"		aux = dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];\n"
/*604*/		"		aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*605*/		"		soma += aux;\n"
/*606*/		"	}\n"
/*607*/		"	da[m] = soma;\n"
/*608*/		"}\n"
/*609*/		"\n"
/*610*/		"//padding.h\n"
/*611*/		"kV paddingfeed(Vector in,Vector out,\n"
/*612*/		"			   int txi,int tyi,\n"
/*613*/		"			   int txo,int tyo,\n"
/*614*/		"			   int t, int l ,\n"
/*615*/		"			   int k0){\n"
/*616*/		"	int k = get_global_id(0) + k0;\n"
/*617*/		"	int x, y, z;\n"
/*618*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*619*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*620*/		"	out[s] = in[k];\n"
/*621*/		"}\n"
/*622*/		"kV paddingBack(Vector gradNext,Vector gradin,\n"
/*623*/		"			   int txi,int tyi,\n"
/*624*/		"			   int txo,int tyo,\n"
/*625*/		"			   int t, int l , int k0){\n"
/*626*/		"	int k = get_global_id(0) + k0;\n"
/*627*/		"	int x, y, z;\n"
/*628*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*629*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*630*/		"	gradin[k] = gradNext[s];\n"
/*631*/		"}\n"
/*632*/		"//pool.h\n"
/*633*/		"kV poolativa(Vector entrada, Vector saida, int lenFilter,\n"
/*634*/		"             int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*635*/		"	int k = get_global_id(0) + k0;\n"
/*636*/		"	int x, y, z;\n"
/*637*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*638*/		"\n"
/*639*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*640*/		"	double mval, v;\n"
/*641*/		"	mval = -DBL_MAX;\n"
/*642*/		"	for (int i = 0; i < lenFilter; ++i) {\n"
/*643*/		"		for (int j = 0; j < lenFilter; ++j) {\n"
/*644*/		"			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*645*/		"			if (v > mval)\n"
/*646*/		"				mval = v;\n"
/*647*/		"		}\n"
/*648*/		"	}\n"
/*649*/		"	saida[k] = mval;\n"
/*650*/		"}\n"
/*651*/		"\n"
/*652*/		"\n"
/*653*/		"kV\n"
/*654*/		"poolCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*655*/		"              int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty, int k0) {\n"
/*656*/		"	int k = get_global_id(0) + k0;\n"
/*657*/		"	int x, y;\n"
/*658*/		"	TensorRemap2D(k, x, y, entradaty)\n"
/*659*/		"	double somaErro = 0, testeMax;\n"
/*660*/		"	Range range;\n"
/*661*/		"	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);\n"
/*662*/		"	for (int z = 0; z < entradatz; ++z) {\n"
/*663*/		"		somaErro = 0;\n"
/*664*/		"		for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*665*/		"			for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*666*/		"				testeMax = (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==\n"
/*667*/		"				            saida[TensorMap(i, j, z, saidatx, saidaty)]);\n"
/*668*/		"				somaErro += testeMax * gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*669*/		"			}\n"
/*670*/		"		}\n"
/*671*/		"		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro;\n"
/*672*/		"	}\n"
/*673*/		"}\n"
/*674*/		"\n"
/*675*/		"\n"
/*676*/		"//poolav.h\n"
/*677*/		"kV PoolAvativa(Vector entrada, Vector saida, int lenFilter,\n"
/*678*/		"               int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*679*/		"	int k = get_global_id(0) + k0;\n"
/*680*/		"	int x, y, z;\n"
/*681*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*682*/		"\n"
/*683*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*684*/		"	double soma = 0, v;\n"
/*685*/		"\n"
/*686*/		"	for (int i = 0; i < lenFilter; ++i) {\n"
/*687*/		"		for (int j = 0; j < lenFilter; ++j) {\n"
/*688*/		"			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*689*/		"\n"
/*690*/		"		}\n"
/*691*/		"	}\n"
/*692*/		"	saida[k] = soma / (lenFilter * lenFilter);\n"
/*693*/		"}\n"
/*694*/		"\n"
/*695*/		"\n"
/*696*/		"kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*697*/		"                   int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty,\n"
/*698*/		"                   int k0) {\n"
/*699*/		"	int k = get_global_id(0) + k0;\n"
/*700*/		"	int x, y;\n"
/*701*/		"	TensorRemap2D(k, x, y, entradaty)\n"
/*702*/		"	double somaErro = 0;\n"
/*703*/		"	Range range;\n"
/*704*/		"	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);\n"
/*705*/		"	for (int z = 0; z < entradatz; ++z) {\n"
/*706*/		"		somaErro = 0;\n"
/*707*/		"		for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*708*/		"			for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*709*/		"				somaErro += gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*710*/		"			}\n"
/*711*/		"		}\n"
/*712*/		"		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro/ (lenFilter * lenFilter);\n"
/*713*/		"	}\n"
/*714*/		"}\n"
/*715*/		"\n"
/*716*/		"\n"
/*717*/		"//relu.h\n"
/*718*/		"kV reluativa(Vector entrada, Vector saida, int k0) {\n"
/*719*/		"	int k = get_global_id(0) + k0;\n"
/*720*/		"	double v = entrada[k];\n"
/*721*/		"	if (v < 0)\n"
/*722*/		"		v = 0;\n"
/*723*/		"	saida[k] = v;\n"
/*724*/		"}\n"
/*725*/		"\n"
/*726*/		"kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*727*/		"	int k = get_global_id(0) + k0;\n"
/*728*/		"	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];\n"
/*729*/		"}\n"
/*730*/		"\n"
/*731*/		"//softmax.h\n"
/*732*/		"kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,\n"
/*733*/		"                 int entradaty,\n"
/*734*/		"                 int k0) {\n"
/*735*/		"	int k = get_global_id(0) + k0;\n"
/*736*/		"	int x, y, z;\n"
/*737*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*738*/		"	exponent[k] = exp(entrada[k]);\n"
/*739*/		"	soma[z] += exponent[k];\n"
/*740*/		"}\n"
/*741*/		"\n"
/*742*/		"kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,\n"
/*743*/		"                 int saidatx, int saidaty, int k0) {\n"
/*744*/		"	int k = get_global_id(0) + k0;\n"
/*745*/		"	int x, y, z;\n"
/*746*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*747*/		"	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];\n"
/*748*/		"}\n"
/*749*/		"\n"
/*750*/		"kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*751*/		"	int k = get_global_id(0) + k0;\n"
/*752*/		"	double xi = entrada[k];\n"
/*753*/		"	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];\n"
/*754*/		"}\n"
/*755*/		"\n"
/*756*/		"\n"
;
#endif // KERNELS_H
