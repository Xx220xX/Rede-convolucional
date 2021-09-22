//
// Created by Henrique on 28-Jul-21.
//

#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include"utils/time_utils.h"
#include"utils/dir.h"
#include "utils/defaultkernel.h"

#if (DEBUG_TRAIN == 1)
#define LOG_TRAIN(fmt, ...)printf("Manage train: ");printf(fmt,## __VA_ARGS__);printf("\n");
#define LOG_TRAIN_v(v, fmv, init, end)\
{for(int _i_=init;_i_<end;_i_++){  \
printf(fmv,v[_i_]);                                   \
}\
};
#else
#define LOG_TRAIN(fmt,...)
#define LOG_TRAIN_v(v,fmv,init,end)
#endif


#define CASOS 0
#define HIT_RATE 1
#define MSE 2

#define PROCESS_ID_END 0
#define PROCESS_ID_LOAD 1
#define PROCESS_ID_TRAIN 2
#define PROCESS_ID_FITNES 3

#define CNN_ERROR(format, ...)    getClErrorWithContext(t->cnn->error.error,t->cnn->error.msg,EXCEPTION_MAX_MSG_SIZE,format, ## __VA_ARGS__)
#define EVENT(event, param) if(event)event(param)
#define CHECK_REAL_TIME(real_time) if(real_time){SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);}

void loadImage(ManageTrain *t);

void loadLabels(ManageTrain *t);

void releaseStringsManageTrain(ManageTrain *t);


void loadData(ManageTrain *t) {
	CHECK_REAL_TIME(t->real_time);
	if (!t->cnn || t->cnn->error.error) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	if (t->cnn->size == 0) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	t->imagens = alloc_mem(t->n_images, sizeof(struct Tensor_t));
	t->targets = alloc_mem(t->n_images, sizeof(struct Tensor_t));
	t->labels = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_RAM | TENSOR_CHAR, t->n_images, 1,
						   1, 1, &t->cnn->error, NULL);
	loadImage(t);///
//	printf("cload\n");
//
	loadLabels(t);
//
	EVENT(t->OnloadedImages, t);
	t->process_id = PROCESS_ID_END;

}

void train(ManageTrain *t) {
	CHECK_REAL_TIME(t->real_time);
//	printf("train epoca %d of %d\n", t->epic, t->n_epics);
	int rn = t->can_run;
//	printf("train can run %d , error %d\n", rn, t->cnn->error.error);
	if (t->cnn->error.error) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	EVENT(t->OnInitTrain, t);

	double local_mse = 0;
	double time_init_train;
	double internal_time = t->current_time;
	Tensor input;
	Tensor output;
	char label;
	char cnn_label;


	if (!t->et.tr_acertos_vector) {
		t->et.tr_acertos_vector = alloc_mem(t->n_images2train, sizeof(double));
		t->et.tr_mse_vector = alloc_mem(t->n_images2train, sizeof(double));
	}
	t->et.tr_numero_epocas = t->n_epics;

	for (; t->can_run && !t->cnn->error.error && t->epic < t->n_epics; t->epic++) {
		//		printf("imagem %d of %d\n", t->image, t->n_images);
		if (t->image >= t->n_images2train) {
			t->image = 0;
			t->et.tr_imagem_atual = 0;
			t->sum_acerto = 0;
			t->sum_erro = 0;
		}
		t->et.tr_numero_imagens = t->n_images2train;

		t->et.tr_epoca_atual = t->epic;
		for (; t->can_run && !t->cnn->error.error && t->image < t->n_images2train; t->image++) {
			time_init_train = getms();
			input = t->imagens[t->image];
			output = t->targets[t->image];
			label = t->labels->hostc[t->image];
			CnnCallT(t->cnn, input);
			CnnLearnT(t->cnn, output);
			CnnCalculeError(t->cnn, &local_mse);

			cnn_label = CnnGetIndexMax(t->cnn);

			t->sum_acerto += (cnn_label == label);

			t->sum_erro += local_mse;
			t->et.tr_acertos_vector[t->image] = t->sum_acerto / (t->image + 1.0);
			t->et.tr_mse_vector[t->image] = t->sum_erro / (t->image + 1.0);
			t->et.tr_erro_medio = t->et.tr_mse_vector[t->image];
			t->et.tr_acerto_medio = t->et.tr_acertos_vector[t->image];
			//			LOG_TRAIN("%lf %lf\n",local_mse,t->et.tr_erro_medio)
			t->et.tr_imagem_atual = t->image;
			internal_time += getms() - time_init_train;
			t->current_time = internal_time;
			t->et.tr_time = t->current_time;
		}
		EVENT(t->OnfinishEpic, t);
	}
	EVENT(t->OnfinishTrain, t);
	t->process_id = PROCESS_ID_END;
}


void fitnes(ManageTrain *t) {
	CHECK_REAL_TIME(t->real_time);
	EVENT(t->OnInitFitnes, t);

	double local_mse = 0;
	Tensor input;
	Tensor output;
	char label;
	char cnnLabel;
	double time_init_train;
	double internal_time = t->current_time;
	int localimage = 0;

	t->et.ft_numero_imagens = t->n_images2fitness;
	t->et.ft_numero_classes = t->n_classes;

	if (!t->et.ft_info) {
		t->et.ft_info_coluns = 3 + t->n_classes;
		t->et.ft_info = alloc_mem(t->n_classes * t->et.ft_info_coluns, sizeof(double));
		t->image = 0;
	}
	for (; t->can_run && !t->cnn->error.error && t->image < t->n_images2fitness; t->image++) {
		localimage = t->image + t->n_images2train;
		time_init_train = getms();
		input = t->imagens[localimage];
		output = t->targets[localimage];
		label = t->labels->hostc[localimage];
		CnnCallT(t->cnn, input);
		cnnLabel = CnnGetIndexMax(t->cnn);
		CnnCalculeErrorTWithOutput(t->cnn, output, &local_mse);

		t->et.ft_info[label * t->et.ft_info_coluns + CASOS]++;
		if (cnnLabel == label) {
			t->et.ft_info[label * t->et.ft_info_coluns + HIT_RATE]++;
		} else {
			t->et.ft_info[label * t->et.ft_info_coluns + 3 + cnnLabel]++;
		}
		t->et.ft_info[label * t->et.ft_info_coluns + MSE] += local_mse;

		t->et.ft_imagem_atual = t->image;

		internal_time += getms() - time_init_train;
		t->current_time = internal_time;
		t->et.ft_time = t->current_time;
	}
	EVENT(t->OnfinishFitnes, t);
	t->process_id = PROCESS_ID_END;
}

void loadImage(ManageTrain *t) {
	if (t->cnn->error.error)return;
	// obtem o tamanho de cada imagem
	Tensor entrada = t->cnn->camadas[0]->entrada;
	size_t pixelsByImage = entrada->x * entrada->y * entrada->z;

	FILE *fimage = fopen(t->file_images.d, "rb");
	if (!fimage) {
		t->cnn->error.error = FAILED_LOAD_FILE;
		CNN_ERROR("Imagens nao foram encontradas em %s\n", t->file_images);
		return;
	}
	Tensor tmp = new_Tensor(t->cnn->cl->context, NULL, TENSOR_CHAR | TENSOR_RAM | TENSOR4D, entrada->x, entrada->y, entrada->z, t->n_images, &t->cnn->error, NULL);

	// bytes de cabeçalho
	fread(tmp->host, 1, t->headers_images, fimage);
	// le as imagens;
	fread(tmp->host, pixelsByImage, t->n_images, fimage);
	fclose(fimage);
	Tensor imageInt = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR_CHAR | TENSOR4D | TENSOR_CPY,
								 tmp->x, tmp->y, tmp->z, tmp->w,
								 &t->cnn->error, tmp->host
	);

	releaseTensor(&tmp);
	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor imageDouble = new_Tensor(t->cnn->cl->context, t->cnn->queue, TENSOR4D,
									imageInt->x, imageInt->y, imageInt->z, imageInt->w,
									&t->cnn->error, NULL
	);


	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao criar tensores imageDouble: ");
		releaseTensor(&imageInt);
		return;
	}


	double value = 255;
	kernel_run_recursive(t->cnn->error.error, t->cnn->kerneldivInt, t->cnn->queue, pixelsByImage * t->n_images,
						 t->cnn->cl->maxworks,
						 K_ARG imageInt->data,
						 K_ARG imageDouble->data,
						 K_ARG value
	);
	synchronizeKernel(t->cnn->queue);

	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao iniciar Kernel: ");
		releaseTensor(&imageInt);
		releaseTensor(&imageDouble);
		return;
	}
	releaseTensor(&imageInt);

	for (int i = 0; i < t->n_images; ++i) {
		t->imagens[i] = new_Tensor(t->cnn->cl->context, t->cnn->queue, 0,
								   imageDouble->x, imageDouble->y, imageDouble->z, 1,
								   &t->cnn->error, NULL
		);
		t->et.ld_imagem_atual = i;
		TensorCpy(t->cnn->queue, t->imagens[i], imageDouble, i);
	}
	releaseTensor(&imageDouble);
	if (t->cnn->error.error) {
		CNN_ERROR("Falha enquanto roda o kernel: ");
	}
}

void loadLabels(ManageTrain *t) {
	if (t->cnn->error.error)return;
	FILE *flabel = fopen(t->file_labels.d, "rb");
	if (!flabel) {
		t->cnn->error.error = FAILED_LOAD_FILE;
		CNN_ERROR("Imagens nao foram encontradas em %s\n", t->file_labels);
		return;
	}
	// bytes de cabeçalho
	fread(t->labels->host, 1, t->headers_labels, flabel);
	// le as imagens;
	fread(t->labels->host, 1, t->n_images, flabel);
	fclose(flabel);
	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor labelInt = new_Tensor(t->cnn->cl->context, t->cnn->queue,
								 TENSOR_CHAR | TENSOR_CPY,
								 t->n_images, 1, 1, 1,
								 &t->cnn->error, t->labels->host);
	Tensor labelDouble = new_Tensor(t->cnn->cl->context, t->cnn->queue,
									TENSOR4D,
									1, t->n_classes, 1, t->n_images,
									&t->cnn->error, NULL);

	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao criar tensores imageDouble: ");
		return;
	}

	kernel_run_recursive(t->cnn->error.error, t->cnn->kernelInt2Vector, t->cnn->queue,
						 t->n_images * t->n_classes, t->cnn->cl->maxworks,
						 K_ARG labelInt->data,
						 K_ARG labelDouble->data,
						 K_ARG t->n_classes
	);
	synchronizeKernel(t->cnn->queue);
	releaseTensor(&labelInt);
	if (t->cnn->error.error) {
		CNN_ERROR("Falha ao iniciar kernel: ");
		releaseTensor(&labelDouble);
		return;
	}
	for (int i = 0; i < t->n_images; ++i) {
		t->targets[i] = new_Tensor(t->cnn->cl->context, t->cnn->queue, 0, t->n_classes, 1, 1, 1, &t->cnn->error, NULL);
		t->cnn->error.error |= TensorCpy(t->cnn->queue, t->targets[i], labelDouble, i);
		t->et.ll_imagem_atual = i;
	}
	releaseTensor(&labelDouble);


}

void releaseEstatitica(Estatistica *et) {
	if (et->tr_acertos_vector)free_mem(et->tr_acertos_vector);
	if (et->tr_mse_vector)free_mem(et->tr_mse_vector);
	if (et->ft_info)free_mem(et->ft_info);
}

int releaseProcess(HANDLE *p_th) {
	if (!p_th)return 1;
	HANDLE process = *p_th;
	if (!process)return 2;
	ThreadKill(process, -1);
	ThreadClose(process);
	*p_th = NULL;
	return 0;
}

void releaseManageTrain(ManageTrain *t) {
	t->can_run = 0;
	releaseProcess(&t->process);
	releaseCnn(&t->cnn);
	if (t->imagens) {
		for (int i = 0; i < t->n_images; i++) {
			releaseTensor(&t->imagens[i]);
		}
		free_mem(t->imagens);
	}
	if (t->targets) {
		for (int i = 0; i < t->n_images; i++) {
			releaseTensor(&t->targets[i]);
		}
		free_mem(t->targets);
	}


	releaseTensor(&t->labels);
	releaseEstatitica(&t->et);
	releaseStringsManageTrain(t);
	*t = (ManageTrain) {0};
	if (t->self_release)
		free_mem(t);

}

void releaseStringsManageTrain(ManageTrain *t) {
	releaseStr(&t->file_images);
	releaseStr(&t->file_labels);
	releaseStr(&t->class_names);
	releaseStr(&t->homePath);
}

void manage2WorkDir(ManageTrain *t) {

	if (!t->cnn || t->cnn->error.error)return;

	if (SetDir(t->homePath.d)) {

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

			{"numero_epocas",  "int",    "Numero de epocas para treinamento"},

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

void loadArgsLuaManageTrain(ManageTrain *t, Dictionary *args) {
	if (t->cnn->error.error)return;
	Dbchar_p arg;
	for (int i = 0; i < args->size; i++) {
		arg = args->values[i];
		if (STREQUALS(arg.name, "work_path")) {
			StrClearAndCopy(t->homePath, arg.value);
		} else if (STREQUALS(arg.name, "file_image")) {
			StrClearAndCopy(t->file_images, arg.value);
		} else if (STREQUALS(arg.name, "file_label")) {
			StrClearAndCopy(t->file_labels, arg.value);
		} else if (STREQUALS(arg.name, "nome_classes")) {
			StrClearAndCopy(t->class_names, arg.value);
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
		} else if (STREQUALS(arg.name, "numero_epocas")) {
			t->n_epics = atoi(arg.value);
		}
	}
}

ManageTrain createManageTrain(char *luafile, double tx_aprendizado, double momento, double decaimento,int luaIsProgram) {
	ManageTrain result = {0};
	result.cnn = createCnnWithWrapperProgram(default_kernel, (Params) {tx_aprendizado, momento, decaimento}, 0, 0, 0, CL_DEVICE_TYPE_GPU);
	LuaputHelpFunctionArgs(helpArgumentsManageTrain);
	if (!luaIsProgram)
		CnnLuaLoadFile(result.cnn, luafile);
	else
		CnnLuaLoadString(result.cnn, luafile);

	loadArgsLuaManageTrain(&result, &result.cnn->luaArgs);

	return result;
}

int ManageTrainloadImages(ManageTrain *t, int runBackground) {
	t->can_run = 1;
	t->real_time = 1;
	t->process_id = PROCESS_ID_LOAD;
	releaseProcess(&t->process);
	if (runBackground) {
		t->process = newThread(loadData, t);
		return t->process != NULL;
	}
	loadData(t);
	return 0;
}

int ManageTraintrain(ManageTrain *t, int runBackground) {
	t->can_run = 1;
	t->real_time = 1;
	t->process_id = PROCESS_ID_TRAIN;
	releaseProcess(&t->process);
	if (runBackground) {
		t->process = newThread(train, t);
		return t->process != NULL;
	}
	train(t);
	return 0;
}

int ManageTrainfitnes(ManageTrain *t, int runBackground) {
	t->can_run = 1;
	t->real_time = 1;
	t->process_id = PROCESS_ID_FITNES;
	releaseProcess(&t->process);
	if (runBackground) {
		t->process = newThread(fitnes, t);
		return t->process != NULL;
	}
	fitnes(t);
	return 0;
}

void waitEndProces(ManageTrain *t) {
	while (t->process_id != PROCESS_ID_END) {
		Sleep(100);
		printf("wainting %d\n", t->process_id);
	}
}

void __manageTrainloop__(ManageTrain *t) {
	ManageEvent ev = NULL;
	int id = t->process_id;
	switch (id) {
		case PROCESS_ID_FITNES:
			ev = t->UpdateFitnes;
			break;
		case PROCESS_ID_TRAIN:
			ev = t->UpdateTrain;
			break;
		case PROCESS_ID_LOAD:
			ev = t->UpdateLoad;
			break;
		case PROCESS_ID_END:
			ev = (ManageEvent) waitEndProces;
			break;
		default:
			printf("Evento nao encontrado\n");
			exit(-1);

	}
	if (!ev)return;
	while (t->process_id != PROCESS_ID_END) {
		EVENT(ev, t);
		Sleep(10);
	}
	EVENT(ev, t);
	EVENT(t->OnFinishLoop,t);
}

void manageTrainLoop(ManageTrain *t, int run_background) {

	if (run_background) {
		if (t->update_loop) {
			ThreadKill(t->update_loop, -1);
			ThreadClose(t->update_loop);
		}
		t->update_loop = newThread(__manageTrainloop__, t);
		return;
	}
	__manageTrainloop__(t);
}
