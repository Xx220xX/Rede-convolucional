#include "utils/manageTrain.h"
#include "cnn/utils/defaultkernel.h"
#include "utils/vectorUtils.h"
// call backs
void onLoad   (ManageTrain *t);

void helpArguments();

void loadArgsLua(ManageTrain *t, List_args *args);

int main(int arg, char **args) {
	system("chcp 65001");
	printf("##############################\n");
	printf("Gabriela IA\n");
	printf("email: gab.cnn.ia@gmail.com\n");
	printf("Versão %s\n", getVersion());
	printf("##############################\n");
	ManageTrain manageTrain = {0};

	manageTrain.cnn = createCnnWithWrapperProgram(default_kernel, (Params) {0.1, 0, 0},
	                                              32, 32, 3, CL_DEVICE_TYPE_GPU);
	LuaputHelpFunctionArgs(helpArguments);

	manageTrain.OnloadedImages = (ManageEvent) onLoad;

	if (arg == 1) {
		CnnLuaConsole(manageTrain.cnn);
	} else if (arg == 2) {
		CnnLuaLoadFile(manageTrain.cnn, args[1]);
	}
	if (!manageTrain.cnn->error.error) {
		int err = manageTrain.cnn->error.error;
		releaseManageTrain(&manageTrain);
		return err;
	}
	manage2WorkDir(&manageTrain);
	loadImages(&manageTrain);
	if (manageTrain.cnn->error.error) {
		printf("%s\n", manageTrain.cnn->error.msg);
	}
	releaseManageTrain(&manageTrain);
	return 0;
}

void helpArguments() {
	struct {
		char *name;
		char *type;
		char *desc;
	} args[] = {
			{"work_path",      "string", "diretorio de trabalho"},
			{"file_image",     "string", "nome do arquivo contendo imagens com endereço relativo a 'work_path'"},
			{"file_label",     "string", "nome do arquivo contendo descrição das imagens com endereço relativo a 'work_path'"},
			{"header_image",   "int",    "número de bytes de cabeçalho no arquivo de imagens"},
			{"header_label",   "int",    "número de bytes de cabeçalho no arquivo de labels"},
			{"numero_imagens", "int",    "número de imagens a serem lidas"},
			{"numero_treino",  "int",    "número de imagens a serem treinada, deve ser menor que 'numero_imagens'"},
			{"numero_fitnes",  "int",    "número de imagens a serem testadas, deve ser menor que 'numero_imagens'-'numero_treino'"},
			{"numero_classes", "int",    "número de de classes possivels no treinamento, deve ser menor que 255 (limite para 8 bits)"},
			{"sep",            "char",   "por padrão é ' '"},
			{"nome_classes",   "string", "nomeda das classes separados por 'sep'"},
			{0}
	};
	printf("Argumentos :\n");
	for (int i = 0; args[i].name; i++) {
		printf("Nome: '%s'\n", args[i].name);
		printf("tipo: '%s'\n", args[i].type);
		printf("descrição: '%s'\n", args[i].desc);
		printf("\n");
	}

}

void loadArgsLua(ManageTrain *t, List_args *args) {
	Dbchar_p arg;
	if (!t->releaseStrings) {
		fflush(stdout);
		fprintf(stderr, "As strings não podem ser estaticas para esta função");
		fflush(stderr);
		releaseManageTrain(t);
		exit(-1);
	}
	for (int i = 0; i < args->size; i++) {
		arg = args->values[i];
		if (STREQUALS(arg.name, "work_path")) {
			check_free_mem(t->homePath);
			t->homePath = copystr(arg.value);
		} else if (STREQUALS(arg.name, "file_image")) {
			check_free_mem(t->file_images);
			t->file_images = copystr(arg.value);
		} else if (STREQUALS(arg.name, "file_label")) {
			check_free_mem(t->file_labels);
			t->file_labels = copystr(arg.value);
		} else if (STREQUALS(arg.name, "nome_classes")) {
			check_free_mem(t->class_names);
			t->class_names = copystr(arg.value);
		} else if (STREQUALS(arg.name, "sep")) {
			t->character_sep = arg.value[0];
		} else if (STREQUALS(arg.name, "header_image")) {
			t->headers_images = atoi(arg.value);
		} else if (STREQUALS(arg.name, "header_label")) {
			t->headers_labels = atoi(arg.value);
		} else if (STREQUALS(arg.name, "numero_imagens")) {
			t->n_images = atoi(arg.value);
		} else if (STREQUALS(arg.name, "numero_treino")) {
			t->n_images2train = atoi(arg.value);
		} else if (STREQUALS(arg.name, "numero_fitnes")) {
			t->n_images2fitness = atoi(arg.value);
		} else if (STREQUALS(arg.name, "numero_classes")) {
			t->n_classes = atoi(arg.value);
		}
	}
}

void printALLIMGS(ManageTrain *t) {
	fflush(stderr);
	fflush(stdout);
	char name[100];
	int len;
	double *v;
	char *aux;
	for (int w = 0; w < t->imagens->w && w < 300; w++) {
		len = snprintf(name, 100, "l%d_%d[", w, (int) ((char *) t->labels->host)[0]);

		v = t->targets->host + w * t->targets->y * sizeof(double);
		aux = name;
		for (int i = 0; i < t->n_classes; i++) {
			aux = aux + len;
			len = snprintf(aux, 100 - (size_t) aux + (size_t) name, "%d_", (int) v[i]);
		}
		aux = aux + len;
		len = snprintf(aux, 100 - (size_t) aux + (size_t) name, "].ppm");
		salveTensor4DAsPPM3(name, t->imagens, t->cnn, w);
	}
}
