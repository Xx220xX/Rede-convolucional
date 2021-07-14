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
/*58*/		"\n"
/*59*/		"\n"
/*60*/		"\n"
/*61*/		"\n"
/*62*/		"kV\n"
/*63*/		"normalizeVector(Vector input, Vector saida, double multiplicador, double somador, double subtrator,\n"
/*64*/		"                int k0) {\n"
/*65*/		"	int k = get_global_id(0) + k0;\n"
/*66*/		"	saida[k] = (input[k] + somador) * multiplicador - subtrator;\n"
/*67*/		"}\n"
/*68*/		"\n"
/*69*/		"\n"
/*70*/		"kV sub(Vector grad, Vector saida, Vector target, int k0) {\n"
/*71*/		"	int k = get_global_id(0) + k0;\n"
/*72*/		"	grad[k] = saida[k] - target[k];\n"
/*73*/		"}\n"
/*74*/		"\n"
/*75*/		"kV div(Vector v, double value, int k0) {\n"
/*76*/		"	int k = get_global_id(0) + k0;\n"
/*77*/		"	v[k] = v[k] / value;\n"
/*78*/		"}\n"
/*79*/		"\n"
/*80*/		"kV divIntDo(__global unsigned char *src, Vector v, double value, int k0) {\n"
/*81*/		"	int k = get_global_id(0) + k0;\n"
/*82*/		"	v[k] = ((double) src[k]) / value;\n"
/*83*/		"}\n"
/*84*/		"\n"
/*85*/		"kV int2vector(__global unsigned char *ints, Vector v, int noptiobs, int k0) {\n"
/*86*/		"	int k = get_global_id(0) + k0;\n"
/*87*/		"	for (int j = 0; j < noptiobs; j++) {\n"
/*88*/		"		v[k * noptiobs + j] = (double) (j == ints[k]);\n"
/*89*/		"	}\n"
/*90*/		"}\n"
/*91*/		"\n"
/*92*/		"int normaliza_range(double f, int max, int lim_min) {\n"
/*93*/		"	if (f <= 0)return 0;\n"
/*94*/		"	if (f >= max - 1)return max - 1;\n"
/*95*/		"	if (lim_min) return ceil(f);\n"
/*96*/		"	else return floor(f);\n"
/*97*/		"}\n"
/*98*/		"\n"
/*99*/		"Range mapeia_entrada_saida(int x, int y, int passo, int tamanhoFiltro, int saidatx, int saidaty, int numeroFiltros) {\n"
/*100*/		"	double a = x, b = y;\n"
/*101*/		"	Range r;\n"
/*102*/		"	r.min.x = normaliza_range((a - tamanhoFiltro + 1) / passo, saidatx, 1);\n"
/*103*/		"	r.min.y = normaliza_range((b - tamanhoFiltro + 1) / passo, saidaty, 1);\n"
/*104*/		"	r.min.z = 0;\n"
/*105*/		"\n"
/*106*/		"	r.max.x = normaliza_range(a / passo, saidatx, 0);\n"
/*107*/		"	r.max.y = normaliza_range(b / passo, saidaty, 0);\n"
/*108*/		"	r.max.z = numeroFiltros - 1;\n"
/*109*/		"	return r;\n"
/*110*/		"}\n"
/*111*/		"\n"
/*112*/		"//bathnorm.h\n"
/*113*/		"\n"
/*114*/		"// achar a media\n"
/*115*/		"kV BatchNormMedia(Vector entrada, Vector media,\n"
/*116*/		"                  int entradatx, int entradaty, int k0) {\n"
/*117*/		"	int z = get_global_id(0) + k0;\n"
/*118*/		"	int x, y;\n"
/*119*/		"	double m = 0;\n"
/*120*/		"	for (x = 0; x < entradatx; x++) {\n"
/*121*/		"		for (y = 0; y < entradaty; y++) {\n"
/*122*/		"			m += entrada[TensorMap(x, y, z, entradatx, entradaty)];\n"
/*123*/		"		}\n"
/*124*/		"	}\n"
/*125*/		"	media[z] = m / (double) (entradatx * entradaty);\n"
/*126*/		"}\n"
/*127*/		"\n"
/*128*/		"// achar a diferenca\n"
/*129*/		"kV BatchNormDiferenca(Vector entrada, Vector media,\n"
/*130*/		"                      Vector diferenca,\n"
/*131*/		"                      Vector diferencaquad,\n"
/*132*/		"                      int entradatx, int entradaty, int k0) {\n"
/*133*/		"	int x, y, z;\n"
/*134*/		"	int k = get_global_id(0) + k0;\n"
/*135*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*136*/		"	diferenca[k] = entrada[k] - media[z];\n"
/*137*/		"	diferencaquad[k] = diferenca[k] * diferenca[k];\n"
/*138*/		"}\n"
/*139*/		"\n"
/*140*/		"kV BatchNormVariance(Vector dif, Vector difQuad,\n"
/*141*/		"                     Vector sumdiferenca, Vector variancia,\n"
/*142*/		"                     double episolon, int diftx, int difty,\n"
/*143*/		"                     int k0) {\n"
/*144*/		"	int z = get_global_id(0) + k0;\n"
/*145*/		"	double sum = 0;\n"
/*146*/		"	double sumdif = 0;\n"
/*147*/		"	for (int x = 0; x < diftx; x++) {\n"
/*148*/		"		for (int y = 0; y < difty; y++) {\n"
/*149*/		"			sum += difQuad[TensorMap(x, y, z, diftx, difty)];\n"
/*150*/		"			sumdif += dif[TensorMap(x, y, z, diftx, difty)];\n"
/*151*/		"		}\n"
/*152*/		"	}\n"
/*153*/		"	sumdiferenca[z] = sumdif;\n"
/*154*/		"	variancia[z] = sqrt(sum / (difty * diftx) + episolon);\n"
/*155*/		"}\n"
/*156*/		"\n"
/*157*/		"// normaliza\n"
/*158*/		"kV BatchNormNormaliza(Vector saida,\n"
/*159*/		"                      Vector norma,\n"
/*160*/		"                      Vector diferenca,\n"
/*161*/		"                      Vector variancia,\n"
/*162*/		"                      Vector Y,\n"
/*163*/		"                      Vector B,\n"
/*164*/		"                      int diferencatx, int diferencaty, int k0) {\n"
/*165*/		"	int x, y, z;\n"
/*166*/		"	int k = get_global_id(0) + k0;\n"
/*167*/		"	TensorRemap(k, x, y, z, diferencatx, diferencaty)\n"
/*168*/		"	norma[k] = diferenca[k] / variancia[z];\n"
/*169*/		"	saida[k] = norma[k] * Y[z] + B[z];\n"
/*170*/		"}\n"
/*171*/		"\n"
/*172*/		"\n"
/*173*/		"kV BatchNormaCalcGrad1(Vector gradIn,\n"
/*174*/		"                       Vector gradNext,\n"
/*175*/		"                       Vector variancia,\n"
/*176*/		"                       Vector media,\n"
/*177*/		"                       Vector Y,\n"
/*178*/		"\n"
/*179*/		"                       Vector somaDif,\n"
/*180*/		"                       Vector entrada,\n"
/*181*/		"                       int entradatx,\n"
/*182*/		"                       int entradaty,\n"
/*183*/		"                       int k0) {\n"
/*184*/		"	int x, y, z;\n"
/*185*/		"	int k = get_global_id(0) + k0;\n"
/*186*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*187*/		"	double M = entradatx * entradaty;\n"
/*188*/		"	double dif_variance = somaDif[z] - entrada[k] + media[z] + (entrada[k] - media[z]) * (M - 1);\n"
/*189*/		"	dif_variance = dif_variance * -1.0 / (variancia[z] * M * M);\n"
/*190*/		"\n"
/*191*/		"	double didx = variancia[z] * (M - 1 / M) + (media[z] - entrada[k]) * dif_variance;\n"
/*192*/		"	didx = didx / (variancia[z] * variancia[z]);\n"
/*193*/		"	didx = didx * gradNext[k];\n"
/*194*/		"	gradIn[k] = didx * Y[z];\n"
/*195*/		"}\n"
/*196*/		"\n"
/*197*/		"kV BatchNormaCalcGrad2(Vector gradNext,\n"
/*198*/		"                       Vector norma,\n"
/*199*/		"                       Vector gradY,\n"
/*200*/		"                       Vector gradB,\n"
/*201*/		"                       int entradatx,\n"
/*202*/		"                       int entradaty,\n"
/*203*/		"                       int k0) {\n"
/*204*/		"	int z = get_global_id(0) + k0;\n"
/*205*/		"	double sumY = 0;\n"
/*206*/		"	double sumB = 0;\n"
/*207*/		"	int k;\n"
/*208*/		"	for (int x = 0; x < entradatx; ++x) {\n"
/*209*/		"		for (int y = 0; y < entradaty; ++y) {\n"
/*210*/		"			k = TensorMap(x, y, z, entradatx, entradaty);\n"
/*211*/		"			sumY += gradNext[k];\n"
/*212*/		"			sumB += gradNext[k] * norma[k];\n"
/*213*/		"		}\n"
/*214*/		"	}\n"
/*215*/		"	gradB[z] = sumB;\n"
/*216*/		"	gradY[z] = sumY;\n"
/*217*/		"}\n"
/*218*/		"\n"
/*219*/		"\n"
/*220*/		"kV batchNormCorrigePeso(Vector gradY,\n"
/*221*/		"                        Vector gradB,\n"
/*222*/		"                        Vector Y,\n"
/*223*/		"                        Vector B,\n"
/*224*/		"                        double hitlearn,\n"
/*225*/		"                        int k0) {\n"
/*226*/		"	int z = get_global_id(0) + k0;\n"
/*227*/		"	B[z] = B[z] - gradB[z] * hitlearn;\n"
/*228*/		"	Y[z] = Y[z] - gradY[z] * hitlearn;\n"
/*229*/		"}\n"
/*230*/		"//conv.h\n"
/*231*/		"//#include\"utils.h\"\n"
/*232*/		"kV convSum(Vector filtro, Vector entrada, Vector saida,\n"
/*233*/		"           int passo, int saidatx, int saidaty, int entradatx, int entradaty,\n"
/*234*/		"           int lenFilter, int entradatz, int k0) {\n"
/*235*/		"	int k = get_global_id(0) + k0;\n"
/*236*/		"	int x, y, filtrok;\n"
/*237*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*238*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*239*/		"	double sum = 0, f, v;\n"
/*240*/		"	for (int m = 0; m < lenFilter; m++)\n"
/*241*/		"		for (int n = 0; n < lenFilter; n++)\n"
/*242*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*243*/		"				f = filtro[TensorMap4D(m, n, z, filtrok, lenFilter, lenFilter, entradatz)];\n"
/*244*/		"				v = entrada[TensorMap(mapeado.x + m, mapeado.y + n, z, entradatx, entradaty)];\n"
/*245*/		"				sum += f * v;\n"
/*246*/		"			}\n"
/*247*/		"	saida[k] = sum;\n"
/*248*/		"\n"
/*249*/		"}\n"
/*250*/		"\n"
/*251*/		"kV convFixWeight(Vector filtro, Vector grad, Vector gradOld, double hitlearn,\n"
/*252*/		"                 double momento, double weightDecay, int k0) {\n"
/*253*/		"	int k = get_global_id(0) + k0;\n"
/*254*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*255*/		"	double w = filtro[k];\n"
/*256*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*257*/		"	gradOld[k] = m;\n"
/*258*/		"}\n"
/*259*/		"\n"
/*260*/		"kV convCalcFiltro(     Vector ds,\n"
/*261*/		"					   Vector entrada,\n"
/*262*/		"					   Vector gradFiltro,\n"
/*263*/		"                       int gradFiltro_tx,\n"
/*264*/		"                       int gradFiltro_ty,\n"
/*265*/		"                       int gradFiltro_tz,\n"
/*266*/		"                       int entrada_tx,\n"
/*267*/		"                       int entrada_ty,\n"
/*268*/		"                       int saida_tx,\n"
/*269*/		"                       int saida_ty,\n"
/*270*/		"                       int passo,\n"
/*271*/		"                       int k0) {\n"
/*272*/		"	int k = get_global_id(0) + k0;\n"
/*273*/		"	int m, n, z, l;\n"
/*274*/		"//	printf(\"kernel %d\\n\",k);\n"
/*275*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*276*/		"	double soma = 0,aux;\n"
/*277*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*278*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*279*/		"			aux = entrada[TensorMap(i*passo+m, j*passo+n,z,entrada_tx,entrada_ty)]\n"
/*280*/		"				   *ds[TensorMap(i,j,l,saida_tx,saida_ty)];\n"
/*281*/		"			//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*282*/		"			soma +=aux;\n"
/*283*/		"		}\n"
/*284*/		"	}\n"
/*285*/		"	gradFiltro[k] = soma;\n"
/*286*/		"}\n"
/*287*/		"\n"
/*288*/		"kV convCalcGrads(Vector filtro,\n"
/*289*/		"				 Vector entrada,\n"
/*290*/		"                 Vector gradEntrada,\n"
/*291*/		"                 Vector gradNext,\n"
/*292*/		"                 int lenFilter,\n"
/*293*/		"                 int filtroz,\n"
/*294*/		"                 int passo,\n"
/*295*/		"                 int entradatx,\n"
/*296*/		"                 int entradaty,\n"
/*297*/		"                 int saidatx,\n"
/*298*/		"                 int saidaty,\n"
/*299*/		"                 int numFilters,\n"
/*300*/		"                 int k0) {\n"
/*301*/		"	int k = get_global_id(0) + k0;\n"
/*302*/		"	int x, y, z;\n"
/*303*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*304*/		"	Range range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, numFilters);\n"
/*305*/		"	int minX, minY;\n"
/*306*/		"	double somaErro = 0, pesoAplicado = 0;\n"
/*307*/		"	double aux;\n"
/*308*/		"	for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*309*/		"		minX = i * passo;\n"
/*310*/		"		for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*311*/		"			minY = j * passo;\n"
/*312*/		"			for (int l = range.min.z; l <= range.max.z; l++) {\n"
/*313*/		"				pesoAplicado = filtro[TensorMap4D(x - minX, y - minY, z, l, lenFilter, lenFilter, filtroz)];\n"
/*314*/		"				aux = pesoAplicado * gradNext[TensorMap(i, j, l, saidatx, saidaty)];\n"
/*315*/		"				//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*316*/		"				somaErro +=aux;\n"
/*317*/		"			}\n"
/*318*/		"		}\n"
/*319*/		"	}\n"
/*320*/		"	gradEntrada[k] = somaErro;\n"
/*321*/		"}\n"
/*322*/		"\n"
/*323*/		"\n"
/*324*/		"//convNc.h\n"
/*325*/		"//#include\"utils.h\"\n"
/*326*/		"kV convncSum(Vector filtro, Vector entrada, Vector saida,\n"
/*327*/		"             int passox, int passoy, int largx,\n"
/*328*/		"             int largy, int saidatx, int saidaty,\n"
/*329*/		"             int entradatx, int entradaty,int fx, int fy,\n"
/*330*/		"             int entradatz, int k0) {\n"
/*331*/		"	int k = get_global_id(0) + k0;\n"
/*332*/		"	int x, y, filtrok;\n"
/*333*/		"	TensorRemap(k, x, y, filtrok, saidatx, saidaty)\n"
/*334*/		"	Ponto3d mapeado = {x * passox, y * passoy, 0};\n"
/*335*/		"	double sum = 0, f, v;\n"
/*336*/		"	for (int i = 0; i < fx; i++)\n"
/*337*/		"		for (int j = 0; j < fy; j++)\n"
/*338*/		"			for (int z = 0; z < entradatz; z++) {\n"
/*339*/		"				f = filtro[TensorMap4D(i, j, z, filtrok, fx, fy, entradatz)];\n"
/*340*/		"				v = entrada[TensorMap(mapeado.x + i * largx, mapeado.y + j * largy, z, entradatx, entradaty)];\n"
/*341*/		"\n"
/*342*/		"				sum += f * v;\n"
/*343*/		"			}\n"
/*344*/		"	saida[k] = sum;\n"
/*345*/		"}\n"
/*346*/		"\n"
/*347*/		"kV convncFixWeight(Vector filtro, Vector grad, Vector gradOld,\n"
/*348*/		"				   double hitlearn,\n"
/*349*/		"                   double momento, double weightDecay, int k0) {\n"
/*350*/		"	int k = get_global_id(0) + k0;\n"
/*351*/		"	double m = grad[k] + gradOld[k] * momento;\n"
/*352*/		"	double w = filtro[k];\n"
/*353*/		"	filtro[k] = w - hitlearn * (m + w * weightDecay);\n"
/*354*/		"	gradOld[k] = m;\n"
/*355*/		"}\n"
/*356*/		"\n"
/*357*/		"kV convncCalcFiltro(Vector ds,\n"
/*358*/		"                    Vector entrada,\n"
/*359*/		"                    Vector gradFiltro,\n"
/*360*/		"                    int gradFiltro_tx,\n"
/*361*/		"                    int gradFiltro_ty,\n"
/*362*/		"                    int gradFiltro_tz,\n"
/*363*/		"\n"
/*364*/		"                    int entrada_tx,\n"
/*365*/		"                    int entrada_ty,\n"
/*366*/		"\n"
/*367*/		"                    int saida_tx,\n"
/*368*/		"                    int saida_ty,\n"
/*369*/		"\n"
/*370*/		"                    int passox,\n"
/*371*/		"                    int passoy,\n"
/*372*/		"\n"
/*373*/		"                    int largx,\n"
/*374*/		"                    int largy,\n"
/*375*/		"                    int k0) {\n"
/*376*/		"	int k = get_global_id(0) + k0;\n"
/*377*/		"	int m, n, z, l;\n"
/*378*/		"	TensorRemap4D(k, m, n, z, l, gradFiltro_tx, gradFiltro_ty, gradFiltro_tz)\n"
/*379*/		"	double soma = 0,aux;\n"
/*380*/		"	for (int i = 0; i < saida_tx; ++i) {\n"
/*381*/		"		for (int j = 0; j < saida_ty; ++j) {\n"
/*382*/		"			aux = entrada[TensorMap(i * passox + m * largx, j * passoy + n * largy, z, entrada_tx, entrada_ty)]\n"
/*383*/		"			        * ds[TensorMap(i, j, l, saida_tx, saida_ty)];\n"
/*384*/		"			//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*385*/		"			soma += aux;\n"
/*386*/		"		}\n"
/*387*/		"	}\n"
/*388*/		"	gradFiltro[k] = soma;\n"
/*389*/		"}\n"
/*390*/		"\n"
/*391*/		"/**\n"
/*392*/		" * equacao a ser implementada\n"
/*393*/		" * x = s*p + m*w\n"
/*394*/		" * onde:\n"
/*395*/		" * 	x é da entrada \n"
/*396*/		" * 	s é da saida\n"
/*397*/		" * 	m é do filtro\n"
/*398*/		" * 	s = (x - m*w)/p\n"
/*399*/		" */\n"
/*400*/		"kV convncCalcGrads(Vector filtro,\n"
/*401*/		"                   Vector entrada,\n"
/*402*/		"                   Vector gradEntrada,\n"
/*403*/		"                   Vector gradNext,\n"
/*404*/		"\n"
/*405*/		"                   int passox,\n"
/*406*/		"                   int passoy,\n"
/*407*/		"                   int largx,\n"
/*408*/		"                   int largy,\n"
/*409*/		"\n"
/*410*/		"                   int entradatx,\n"
/*411*/		"                   int entradaty,\n"
/*412*/		"                   int saidatx,\n"
/*413*/		"                   int saidaty,\n"
/*414*/		"\n"
/*415*/		"                   int fx,\n"
/*416*/		"                   int fy,\n"
/*417*/		"                   int fz,\n"
/*418*/		"                   int numFilters,\n"
/*419*/		"\n"
/*420*/		"                   int k0) {\n"
/*421*/		"	int k = get_global_id(0) + k0;\n"
/*422*/		"	int x, y, z;\n"
/*423*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*424*/		"	Range range_filtro ;\n"
/*425*/		"	range_filtro.min.x = 0;\n"
/*426*/		"	if ((entradatx - x - (fx - 1) * largx) < 0) {\n"
/*427*/		"		range_filtro.min.x = -entradatx + x + fx;\n"
/*428*/		"	}\n"
/*429*/		"	range_filtro.max.x = fx - 1;\n"
/*430*/		"	if (x - (fx - 1) * largx < 0) {\n"
/*431*/		"		range_filtro.max.x = x / largx;\n"
/*432*/		"	}\n"
/*433*/		"	range_filtro.min.y = 0;\n"
/*434*/		"	if ((entradaty - y - (fy - 1) * largy) < 0) {\n"
/*435*/		"		range_filtro.min.y = -entradaty + y + fy;\n"
/*436*/		"	}\n"
/*437*/		"	range_filtro.max.y = fy - 1;\n"
/*438*/		"	if (y - (fy - 1) * largy < 0) {\n"
/*439*/		"		range_filtro.max.y = y / largy;\n"
/*440*/		"	}\n"
/*441*/		"	int sx, sy;\n"
/*442*/		"	double somaErro = 0,aux, pesoAplicado = 0;\n"
/*443*/		"	for (int m = range_filtro.min.x; m <= range_filtro.max.x; m++) {\n"
/*444*/		"		sx = (x - m * largx) / passox;\n"
/*445*/		"		if (sx * passox + m * largx != x)continue;\n"
/*446*/		"		for (int n = range_filtro.min.y; n <= range_filtro.max.y; n++) {\n"
/*447*/		"			sy = (y - n * largy) / passox;\n"
/*448*/		"			if (sy * passoy + n * largy != y)continue;\n"
/*449*/		"			for (int l = 0; l < fz; l++) {\n"
/*450*/		"				pesoAplicado = filtro[TensorMap4D(m, n, z, l, fx, fy, fz)];\n"
/*451*/		"				aux = pesoAplicado * gradNext[TensorMap(sx, sy, l, saidatx, saidaty)];\n"
/*452*/		"				//aux = (!(isnan(aux) || isinf(aux)))*aux;\n"
/*453*/		"				somaErro +=aux;\n"
/*454*/		"			}\n"
/*455*/		"		}\n"
/*456*/		"	}\n"
/*457*/		"	gradEntrada[k] = somaErro;\n"
/*458*/		"}\n"
/*459*/		"\n"
/*460*/		"\n"
/*461*/		"//dropout.h\n"
/*462*/		"#define MAX_INT_DP  ((1UL << 31) - 1)\n"
/*463*/		"long randoml(unsigned long seed,unsigned long id) {\n"
/*464*/		"	seed += id;\n"
/*465*/		"	return (seed * 0x5deece66dL + 0xbL) & MAX_INT_DP;\n"
/*466*/		"}\n"
/*467*/		"\n"
/*468*/		"double randomD(unsigned long seed,unsigned long id) {\n"
/*469*/		"	return (double) randoml(seed, id) / (double) MAX_INT_DP;\n"
/*470*/		"}\n"
/*471*/		"\n"
/*472*/		"kV dropativa(Vector entrada, Vector saida, __global char *hitmap, long seed,\n"
/*473*/		"			 double pativa, int k0) {\n"
/*474*/		"	int i = get_global_id(0) + k0;\n"
/*475*/		"//	printf(\"kernel %lf %lf %g %g\\n\",randomD(seed, i),pativa,(double)(seed +i),(double)MAX_INT_DP);\n"
/*476*/		"	char teste = (char) (randomD(seed, i) <= pativa);\n"
/*477*/		"	hitmap[i] = teste;\n"
/*478*/		"	saida[i] = teste * entrada[i];\n"
/*479*/		"}\n"
/*480*/		"\n"
/*481*/		"\n"
/*482*/		"kV dropcalcgrad(Vector gradentrada, __global char *hitmap, Vector gradnext, int k0) {\n"
/*483*/		"	int i = get_global_id(0) + k0;\n"
/*484*/		"	gradentrada[i] = hitmap[i] * gradnext[i];\n"
/*485*/		"}\n"
/*486*/		"\n"
/*487*/		"//fullconnect.h\n"
/*488*/		"double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }\n"
/*489*/		"\n"
/*490*/		"double difsigmoid(double x) {\n"
/*491*/		"	double tmp = sigmoid(x);\n"
/*492*/		"	return tmp * (1.0 - tmp);\n"
/*493*/		"}\n"
/*494*/		"\n"
/*495*/		"double tanghG(double x) { return tanh(x); }\n"
/*496*/		"\n"
/*497*/		"double diftanhG(double x) {\n"
/*498*/		"	double tmp = tanh(x);\n"
/*499*/		"	return (1.0 - tmp * tmp);\n"
/*500*/		"}\n"
/*501*/		"\n"
/*502*/		"double relu(double x) { return x > 0 ? x : 0.0; }\n"
/*503*/		"\n"
/*504*/		"double difrelu(double x) { return x > 0 ? 1.0 : 0.0; }\n"
/*505*/		"\n"
/*506*/		"double func(int id, double x) {\n"
/*507*/		"	switch (id) {\n"
/*508*/		"		case 0:\n"
/*509*/		"			return sigmoid(x);\n"
/*510*/		"		case 1:\n"
/*511*/		"			return difsigmoid(x);\n"
/*512*/		"		case 2:\n"
/*513*/		"			return tanghG(x);\n"
/*514*/		"		case 3:\n"
/*515*/		"			return diftanhG(x);\n"
/*516*/		"		case 4:\n"
/*517*/		"			return relu(x);\n"
/*518*/		"		case 5:\n"
/*519*/		"			return difrelu(x);\n"
/*520*/		"		default:\n"
/*521*/		"			return 0;\n"
/*522*/		"	}\n"
/*523*/		"}\n"
/*524*/		"\n"
/*525*/		"kV fullfeed(Vector entrada, Vector pesos, Vector z, Vector saida,\n"
/*526*/		"			int funcaoativacao, int inx, int iny, int inz, int pesosx, int pesosy, int k0) {\n"
/*527*/		"	int m = get_global_id(0) + k0;\n"
/*528*/		"	double valorEntrada = 0;\n"
/*529*/		"	int n;\n"
/*530*/		"	for (n = 0; n < pesosy; n++) {\n"
/*531*/		"		valorEntrada += entrada[n] * pesos[TensorMap(m, n, 0, pesosx, pesosy)];\n"
/*532*/		"	}\n"
/*533*/		"	z[m] = valorEntrada;\n"
/*534*/		"	saida[m] = func(funcaoativacao, valorEntrada);\n"
/*535*/		"}\n"
/*536*/		"\n"
/*537*/		"kV fullfixweight(Vector a,\n"
/*538*/		"			  Vector pesos,\n"
/*539*/		"			  Vector dw,\n"
/*540*/		"			  Vector dz,\n"
/*541*/		"			  double hitlearn,\n"
/*542*/		"			  double decaimentoDePeso,\n"
/*543*/		"			  double momento,\n"
/*544*/		"			  int pesosy,\n"
/*545*/		"			  int k0) {\n"
/*546*/		"	int k = get_global_id(0) + k0;\n"
/*547*/		"	int m, n;\n"
/*548*/		"	m = k / pesosy;\n"
/*549*/		"	n = k % pesosy;\n"
/*550*/		"	dw[k] = dz[m] * a[n] + dw[k] * momento;\n"
/*551*/		"	pesos[k] = pesos[k] - hitlearn * (dw[k] + pesos[k] * decaimentoDePeso);\n"
/*552*/		"}\n"
/*553*/		"\n"
/*554*/		"kV fullcalcgrads1(Vector dz, Vector ds, Vector z, int dfa, int k0) {\n"
/*555*/		"	int m = get_global_id(0) + k0;\n"
/*556*/		"	dz[m] = ds[m] * func(dfa, z[m]);\n"
/*557*/		"}\n"
/*558*/		"\n"
/*559*/		"kV fullcalcgrads2(Vector dz, Vector da, Vector pesos, int pesosx, int pesosy,\n"
/*560*/		"				  int k0) {\n"
/*561*/		"	int m = get_global_id(0) + k0;\n"
/*562*/		"	double soma = 0;\n"
/*563*/		"	for (int n = 0; n < pesosx; ++n) {\n"
/*564*/		"		soma += dz[n] * pesos[TensorMap(n, m, 0, pesosx, pesosy)];\n"
/*565*/		"	}\n"
/*566*/		"	da[m] = soma;\n"
/*567*/		"}\n"
/*568*/		"\n"
/*569*/		"//padding.h\n"
/*570*/		"kV paddingfeed(Vector in,Vector out,\n"
/*571*/		"			   int txi,int tyi,\n"
/*572*/		"			   int txo,int tyo,\n"
/*573*/		"			   int t, int l ,\n"
/*574*/		"			   int k0){\n"
/*575*/		"	int k = get_global_id(0) + k0;\n"
/*576*/		"	int x, y, z;\n"
/*577*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*578*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*579*/		"	out[s] = in[k];\n"
/*580*/		"}\n"
/*581*/		"kV paddingBack(Vector gradNext,Vector gradin,\n"
/*582*/		"			   int txi,int tyi,\n"
/*583*/		"			   int txo,int tyo,\n"
/*584*/		"			   int t, int l , int k0){\n"
/*585*/		"	int k = get_global_id(0) + k0;\n"
/*586*/		"	int x, y, z;\n"
/*587*/		"	TensorRemap(k, x, y, z, txi, tyi)\n"
/*588*/		"	int s = TensorMap(x+t,y+l,z,txo,tyo);\n"
/*589*/		"	gradin[k] = gradNext[s];\n"
/*590*/		"}\n"
/*591*/		"//pool.h\n"
/*592*/		"kV poolativa(Vector entrada, Vector saida, int lenFilter,\n"
/*593*/		"             int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*594*/		"	int k = get_global_id(0) + k0;\n"
/*595*/		"	int x, y, z;\n"
/*596*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*597*/		"\n"
/*598*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*599*/		"	double mval, v;\n"
/*600*/		"	mval = -DBL_MAX;\n"
/*601*/		"	for (int i = 0; i < lenFilter; ++i) {\n"
/*602*/		"		for (int j = 0; j < lenFilter; ++j) {\n"
/*603*/		"			v = entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*604*/		"			if (v > mval)\n"
/*605*/		"				mval = v;\n"
/*606*/		"		}\n"
/*607*/		"	}\n"
/*608*/		"	saida[k] = mval;\n"
/*609*/		"}\n"
/*610*/		"\n"
/*611*/		"\n"
/*612*/		"kV\n"
/*613*/		"poolCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*614*/		"              int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty, int k0) {\n"
/*615*/		"	int k = get_global_id(0) + k0;\n"
/*616*/		"	int x, y;\n"
/*617*/		"	TensorRemap2D(k, x, y, entradaty)\n"
/*618*/		"	double somaErro = 0, testeMax;\n"
/*619*/		"	Range range;\n"
/*620*/		"	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);\n"
/*621*/		"	for (int z = 0; z < entradatz; ++z) {\n"
/*622*/		"		somaErro = 0;\n"
/*623*/		"		for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*624*/		"			for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*625*/		"				testeMax = (entrada[TensorMap(x, y, z, entradatx, entradaty)] ==\n"
/*626*/		"				            saida[TensorMap(i, j, z, saidatx, saidaty)]);\n"
/*627*/		"				somaErro += testeMax * gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*628*/		"			}\n"
/*629*/		"		}\n"
/*630*/		"		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro;\n"
/*631*/		"	}\n"
/*632*/		"}\n"
/*633*/		"\n"
/*634*/		"\n"
/*635*/		"//poolav.h\n"
/*636*/		"kV PoolAvativa(Vector entrada, Vector saida, int lenFilter,\n"
/*637*/		"               int passo, int saidatx, int saidaty, int entradatx, int entradaty, int k0) {\n"
/*638*/		"	int k = get_global_id(0) + k0;\n"
/*639*/		"	int x, y, z;\n"
/*640*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*641*/		"\n"
/*642*/		"	Ponto3d mapeado = {x * passo, y * passo, 0};\n"
/*643*/		"	double soma = 0, v;\n"
/*644*/		"\n"
/*645*/		"	for (int i = 0; i < lenFilter; ++i) {\n"
/*646*/		"		for (int j = 0; j < lenFilter; ++j) {\n"
/*647*/		"			soma += entrada[TensorMap(mapeado.x + i, mapeado.y + j, z, entradatx, entradaty)];\n"
/*648*/		"\n"
/*649*/		"		}\n"
/*650*/		"	}\n"
/*651*/		"	saida[k] = soma / (lenFilter * lenFilter);\n"
/*652*/		"}\n"
/*653*/		"\n"
/*654*/		"\n"
/*655*/		"kV PoolAvCalcGrads(Vector entrada, Vector gradEntrada, Vector gradNext, Vector saida,\n"
/*656*/		"                   int lenFilter, int passo, int entradatx, int entradaty, int entradatz, int saidatx, int saidaty,\n"
/*657*/		"                   int k0) {\n"
/*658*/		"	int k = get_global_id(0) + k0;\n"
/*659*/		"	int x, y;\n"
/*660*/		"	TensorRemap2D(k, x, y, entradaty)\n"
/*661*/		"	double somaErro = 0;\n"
/*662*/		"	Range range;\n"
/*663*/		"	range = mapeia_entrada_saida(x, y, passo, lenFilter, saidatx, saidaty, 1);\n"
/*664*/		"	for (int z = 0; z < entradatz; ++z) {\n"
/*665*/		"		somaErro = 0;\n"
/*666*/		"		for (int i = range.min.x; i <= range.max.x; i++) {\n"
/*667*/		"			for (int j = range.min.y; j <= range.max.y; j++) {\n"
/*668*/		"				somaErro += gradNext[TensorMap(i, j, z, saidatx, saidaty)];\n"
/*669*/		"			}\n"
/*670*/		"		}\n"
/*671*/		"		gradEntrada[TensorMap(x, y, z, entradatx, entradaty)] = somaErro/ (lenFilter * lenFilter);\n"
/*672*/		"	}\n"
/*673*/		"}\n"
/*674*/		"\n"
/*675*/		"\n"
/*676*/		"//relu.h\n"
/*677*/		"kV reluativa(Vector entrada, Vector saida, int k0) {\n"
/*678*/		"	int k = get_global_id(0) + k0;\n"
/*679*/		"	double v = entrada[k];\n"
/*680*/		"	if (v < 0)\n"
/*681*/		"		v = 0;\n"
/*682*/		"	saida[k] = v;\n"
/*683*/		"}\n"
/*684*/		"\n"
/*685*/		"kV relucalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*686*/		"	int k = get_global_id(0) + k0;\n"
/*687*/		"	gradentrada[k] = entrada[k] <= 0.0 ? (0) : gradnext[k];\n"
/*688*/		"}\n"
/*689*/		"\n"
/*690*/		"//softmax.h\n"
/*691*/		"kV SoftMaxativa1(Vector entrada, Vector exponent, Vector soma, int entradatx,\n"
/*692*/		"                 int entradaty,\n"
/*693*/		"                 int k0) {\n"
/*694*/		"	int k = get_global_id(0) + k0;\n"
/*695*/		"	int x, y, z;\n"
/*696*/		"	TensorRemap(k, x, y, z, entradatx, entradaty)\n"
/*697*/		"	exponent[k] = exp(entrada[k]);\n"
/*698*/		"	soma[z] += exponent[k];\n"
/*699*/		"}\n"
/*700*/		"\n"
/*701*/		"kV SoftMaxativa2(Vector exponet, Vector soma, Vector saida,\n"
/*702*/		"                 int saidatx, int saidaty, int k0) {\n"
/*703*/		"	int k = get_global_id(0) + k0;\n"
/*704*/		"	int x, y, z;\n"
/*705*/		"	TensorRemap(k, x, y, z, saidatx, saidaty)\n"
/*706*/		"	saida[k] = exponet[TensorMap(x, y, z, saidatx, saidaty)] / soma[z];\n"
/*707*/		"}\n"
/*708*/		"\n"
/*709*/		"kV softMaxcalcgrad(Vector gradentrada, Vector entrada, Vector gradnext, int k0) {\n"
/*710*/		"	int k = get_global_id(0) + k0;\n"
/*711*/		"	double xi = entrada[k];\n"
/*712*/		"	gradentrada[k] = xi * (1.0 - xi) * gradnext[k];\n"
/*713*/		"}\n"
/*714*/		"\n"
/*715*/		"\n"
;
#endif // KERNELS_H
