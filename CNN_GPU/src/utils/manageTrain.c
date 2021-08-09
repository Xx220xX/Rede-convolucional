//
// Created by Henrique on 28-Jul-21.
//

#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include"utils/time_utils.h"
#include"utils/dir.h"
#include "utils/defaultkernel.h"
#include "windows.h"

#define ERROR(format, ...)    getClErrorWithContext(t->cnn->error.error, \
t->cnn->error.msg,EXCEPTION_MAX_MSG_SIZE,format, ## __VA_ARGS__)

#define EVENT(event, param) if(event)event(param)

void loadImage(ManageTrain *t);

void loadLabels(ManageTrain *t);

void releaseStringsManageTrain(ManageTrain *t);


void loadImages(ManageTrain *t) {
	if (!t->cnn || t->cnn->error.error)return;
	if (t->cnn->size == 0)return;
	Tensor entrada = t->cnn->camadas[0]->entrada;
	t->imagens = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_HOST | TENSOR_SMEM | TENSOR4D, entrada->x,
							entrada->y, entrada->z, t->n_images, &t->cnn->error, NULL);
	t->labels = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_HOST | TENSOR_SMEM | TENSOR_CHAR, t->n_images, 1,
						   1, 1, &t->cnn->error, NULL);
	t->targets = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_HOST | TENSOR_SMEM, t->n_images, t->n_classes, 1,
							1, &t->cnn->error, NULL);
	loadImage(t);
	loadLabels(t);

	EVENT(t->OnloadedImages, t);

}

void train(ManageTrain *t) {
	EVENT(t->OnInitTrain, t);
	double time_init_train;
	double internal_time = t->current_time;
	double *input;
	double *output;
	char label;
	char cnn_label;
	if (!t->et.tr_acertos_vector) {
		t->et.tr_acertos_vector = alloc_mem(t->n_images2train, sizeof(double));
		t->et.tr_mse_vector = alloc_mem(t->n_images2train, sizeof(double));
	}
	t->et.tr_numero_epocas = t->n_epics;

	for (; t->can_run && !t->cnn->error.error && t->epic < t->n_epics; t->epic++) {
		if (t->image >= t->n_images) {
			t->image = 0;
			t->et.tr_imagem_atual = 0;
			t->et.tr_numero_imagens = t->n_images;
		}
		t->et.tr_epoca_atual = t->epic;
		for (; t->can_run && !t->cnn->error.error && t->image < t->n_images2train; t->image++) {
			time_init_train = getms();
			input = t->imagens->host + (t->imagens->bytes * t->image);
			output = t->targets->host + (t->targets->bytes * t->image);
			label = ((char *) t->labels->host)[t->image];

			CnnCall(t->cnn, input);
			CnnLearn(t->cnn, output);

			CnnCalculeError(t->cnn);
			cnn_label = CnnGetIndexMax(t->cnn);

			t->sum_acerto += cnn_label == label;
			t->sum_erro += t->cnn->normaErro;
			t->et.tr_acertos_vector[t->image] = t->sum_acerto / (t->image + 1.0);
			t->et.tr_mse_vector[t->image] = t->sum_erro / (t->image + 1.0);
			t->et.tr_imagem_atual = t->image;
			internal_time += getms() - time_init_train;
			t->current_time = internal_time;
		}
		EVENT(t->OnfinishEpic, t);
	}
	EVENT(t->OnfinishTrain, t);
}

#define HIT_RATE 0
#define MSE 1
#define CASOS 2

void fitnes(ManageTrain *t) {
	EVENT(t->OnInitFitnes, t);
	double *input;
	double *output;
	char label;
	char cnnLabel;
	int img_atual;
	Tensor info = NULL;
	if (!t->et.ft_info) {
		t->et.ft_info = alloc_mem(t->n_images * 3, sizeof(double));
	}
	info = new_Tensor(NULL,
					  NULL,
					  TENSOR_HOST | TENSOR_UPTR, t->n_classes, 3, 1, 1,
					  &t->cnn->error, t->et.ft_info);
	t->et.ft_numero_imagens = t->n_images2fitness;
	t->et.ft_numero_classes = t->n_classes;

	for (; t->can_run && !t->cnn->error.error && t->image < t->n_images2fitness; t->image++) {
		img_atual = t->image + t->n_images2train;
		input = t->imagens->host + (t->imagens->bytes * img_atual);
		output = t->targets->host + (t->targets->bytes * img_atual);
		label = ((char *) t->labels->host)[img_atual];
		CnnCall(t->cnn, input);
		cnnLabel = CnnGetIndexMax(t->cnn);


		t->et.ft_info[Tensor_Map(info, label, CASOS, 0)]++;
		if (cnnLabel == label) {
			t->et.ft_info[Tensor_Map(info, label, HIT_RATE, 0)]++;
		}
		CnnCalculeErrorWithOutput(t->cnn, output);
		t->et.ft_info[Tensor_Map(info, label, MSE, 0)] += t->cnn->normaErro;
		t->et.ft_imagem_atual = t->image;
	}
	releaseTensor(&info);
	EVENT(t->OnInitFitnes, t);
}

void loadImage(ManageTrain *t) {
	if (t->cnn->error.error)return;
	// obtem o tamanho de cada imagem
	size_t pixelsByImage = t->imagens->x * t->imagens->y * t->imagens->z;

	FILE *fimage = fopen(t->file_images, "rb");
	if (!fimage) {
		t->cnn->error.error = FAILED_LOAD_FILE;
		ERROR("Imagens nao foram encontradas em %s\n", t->file_images);
		return;
	}
	// bytes de cabeçalho
	fread(t->imagens->host, 1, t->headers_images, fimage);
	// le as imagens;
	fread(t->imagens->host, pixelsByImage, t->n_images, fimage);
	fclose(fimage);

	Tensor imageInt = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_CHAR | TENSOR4D | TENSOR_UPTR,
								 t->imagens->x, t->imagens->y, t->imagens->z, t->imagens->w,
								 &t->cnn->error, t->imagens->host
	);

	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor imageDouble = new_Tensor(t->cnn->cl->context, t->cnn->queue,
									TENSOR4D,
									t->imagens->x, t->imagens->y, t->imagens->z, t->imagens->w,
									&t->cnn->error, NULL);

	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageDouble: ");
		return;
	}


	double value = 255;
	kernel_run_recursive(t->cnn->error.error, t->cnn->kerneldivInt, t->cnn->queue, pixelsByImage * t->n_images,
						 t->cnn->cl->maxworks,
						 K_ARG imageInt->data,
						 K_ARG imageDouble->data,
						 K_ARG value
	);
	if (t->cnn->error.error) {
		ERROR("Falha ao iniciar Kernel: ");
		releaseTensor(&imageInt);
		releaseTensor(&imageDouble);
		return;
	}
	releaseTensor(&imageInt);
	t->cnn->error.error = TensorGetValuesMem(t->cnn->queue, imageDouble, t->imagens->host,
											 t->imagens->bytes * t->imagens->w);

	releaseTensor(&imageDouble);
	if (t->cnn->error.error) {
		ERROR("Falha enquanto roda o kernel: ");
	}
}

void loadLabels(ManageTrain *t) {
	if (t->cnn->error.error)return;
	FILE *flabel = fopen(t->file_labels, "rb");
	if (!flabel) {
		t->cnn->error.error = FAILED_LOAD_FILE;
		ERROR("Imagens nao foram encontradas em %s\n", t->file_labels);
		return;
	}
	// bytes de cabeçalho
	fread(t->labels->host, 1, t->headers_labels, flabel);
	// le as imagens;
	fread(t->labels->host, 1, t->n_images, flabel);
	fclose(flabel);

	Tensor labelInt = new_Tensor(t->cnn->cl->context, t->cnn->queue,
								 TENSOR_CHAR | TENSOR_UPTR,
								 t->labels->x, t->labels->y, 1, 1,
								 &t->cnn->error, t->labels->host);
	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor labelDouble = new_Tensor(t->cnn->cl->context, t->cnn->queue,
									0,
									t->targets->x, t->targets->y, 1, 1,
									&t->cnn->error, NULL);

	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageDouble: ");
		return;
	}

	kernel_run_recursive(t->cnn->error.error, t->cnn->kernelInt2Vector, t->cnn->queue,
						 t->labels->x * t->labels->y, t->cnn->cl->maxworks,
						 K_ARG labelInt->data,
						 K_ARG labelDouble->data,
						 K_ARG labelDouble->y
	);
	if (t->cnn->error.error) {
		ERROR("Falha ao iniciar kernel: ");
		releaseTensor(&labelInt);
		releaseTensor(&labelDouble);
		return;
	}

	t->cnn->error.error = TensorGetValuesMem(t->cnn->queue, labelDouble, t->targets->host, t->targets->bytes);
	releaseTensor(&labelInt);
	if (t->cnn->error.error) {
		ERROR("Falha enquanto roda o kernel: ");
	}
	releaseTensor(&labelDouble);
}

void releaseEstatitica(Estatistica *et) {
	if (et->tr_acertos_vector)free_mem(et->tr_acertos_vector);
	if (et->tr_mse_vector)free_mem(et->tr_mse_vector);
	if (et->ft_info)free_mem(et->ft_info);
}

void releaseManageTrain(ManageTrain *t) {
	t->can_run = 0;
	pthread_join(t->process, NULL);
	releaseCnn(&t->cnn);
	releaseTensor(&t->imagens);
	releaseTensor(&t->targets);
	releaseTensor(&t->labels);
	releaseEstatitica(&t->et);
	releaseStringsManageTrain(t);

}

void releaseStringsManageTrain(ManageTrain *t) {
	if (t->releaseStrings) {
		if (t->file_labels)free_mem(t->file_labels);
		if (t->file_images)free_mem(t->file_images);
		if (t->class_names)free_mem(t->class_names);
		if (t->homePath)free_mem(t->class_names);
	}
}

void manage2WorkDir(ManageTrain *t) {
	if (!t->cnn || t->cnn->error.error)return;
	if (SetDir(t->homePath)) {
		t->cnn->error.error = FAILED_SET_DIR;
		return;
	};
}

void manageTrainSetEvent(ManageEvent *dst, ManageEvent src) {
	*dst = src;
}

void manageTrainSetRun(ManageTrain *t, int run) {
	t->can_run = run != 0;
}

void helpArgumentsManageTrain() {
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

void loadArgsLuaManageTrain(ManageTrain *t, List_args *args) {
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

ManageTrain createManageTrain(char *luafile, double tx_aprendizado, double momento, double decaimento) {
	ManageTrain result;

	result.cnn = createCnnWithWrapperProgram(default_kernel, (Params) {tx_aprendizado, momento, decaimento},
											 0, 0, 0, CL_DEVICE_TYPE_GPU);
	LuaputHelpFunctionArgs(helpArgumentsManageTrain);
	CnnLuaLoadFile(result.cnn, luafile);
	loadArgsLuaManageTrain(&result, &result.cnn->luaArgs);
	return result;
}

void ManageTrainInitThreadHigh(ManageTrain *t) {
	int tid=0;
	HANDLE thread = CreateThread(
			NULL,                   // default security attributes
			0,                      // use default stack size
			(LPTHREAD_START_ROUTINE) train,       // thread function name
			t,          // argument to thread function
			0,                      // use default creation flags
			(LPDWORD) &tid);   // returns the thread identifier
			;

}
