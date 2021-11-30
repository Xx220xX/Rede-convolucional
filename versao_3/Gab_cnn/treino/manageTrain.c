//
// Created by Henrique on 28-Jul-21.
//


#include "Manage.h"
#include"dir.h"

#define CASOS 0
#define HIT_RATE 1
#define MSE 2

#define PROCESS_ID_END 0
#define PROCESS_ID_LOAD 1
#define PROCESS_ID_TRAIN 2
#define PROCESS_ID_FITNES 3

#define EVENT(event, param) if(event)event(param)
#define CHECK_REAL_TIME(real_time) if(real_time){SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);}
#define ReleaseStr(string)if(string)free_mem(string);string = NULL

void loadImage(Manage *t);

void loadLabels(Manage *t);

void releaseStringsManageTrain(Manage *t);

double getus() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
	return (double) ret / 10.0;
}

void loadData(Manage *t) {
	CHECK_REAL_TIME(t->real_time);
	if (!t->cnn || t->cnn->erro->error) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	if (t->cnn->l == 0) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	t->imagens = alloc_mem(t->n_images, sizeof(Tensor));
	t->targets = alloc_mem(t->n_images, sizeof(Tensor));
	t->labels = Tensor_new(1, t->n_images, 1, 1, t->cnn->erro, TENSOR_RAM | TENSOR_CHAR);
	loadImage(t);
	loadLabels(t);
	t->process_id = PROCESS_ID_END;
}

void train(Manage *t) {
	CHECK_REAL_TIME(t->real_time);
	if (t->cnn->erro->error) {
		t->process_id = PROCESS_ID_END;
		return;
	}
	REAL local_mse = 0;
	double time_init_train;
	double time_init_epic;
	double internal_time = t->current_time;
	Tensor input;
	Tensor target;
	char label;
	int cnn_label;
	if (!t->et.tr_acertos_vector) {
		t->et.tr_acertos_vector = alloc_mem(t->n_images2train, sizeof(double));
		t->et.tr_mse_vector = alloc_mem(t->n_images2train, sizeof(double));
		t->et.tr_erro_medio = 1;
	}
	t->et.tr_numero_epocas = t->n_epics;
	time_init_train = getus();
	long long int imgs = 0;
	t->et.tr_imps = 0;
	for (; t->can_run && !t->cnn->erro->error && t->epic < t->n_epics; t->epic++) {
		if (t->image >= t->n_images2train) {
			t->image = 0;
			t->et.tr_imagem_atual = 0;
			t->sum_acerto = 0;
		}
		t->et.tr_numero_imagens = t->n_images2train;
		t->et.tr_epoca_atual = t->epic;
		imgs = 0;
		time_init_epic = getus();
		for (; t->can_run && !t->cnn->erro->error && t->image < t->n_images2train; t->image++) {
			imgs++;
			input = t->imagens[t->image];
			target = t->targets[t->image];
			label = ((char *) t->labels->data)[t->image];
			t->cnn->predict(t->cnn, input);
			t->cnn->learn(t->cnn, target);
			local_mse = t->cnn->mse(t->cnn);
			cnn_label = t->cnn->maxIndex(t->cnn);
			t->sum_acerto += (cnn_label == label);
			t->et.tr_acertos_vector[t->image] = t->sum_acerto / (t->image + 1.0);
			t->et.tr_erro_medio = t->et.tr_erro_medio * 0.4 + 0.6 * local_mse;
			t->et.tr_mse_vector[t->image] = t->et.tr_erro_medio;
			t->et.tr_acerto_medio = t->et.tr_acertos_vector[t->image];
			t->current_time = internal_time + getus() - time_init_train;
			t->et.tr_imagem_atual = t->image;
			t->et.tr_time = t->current_time;

			t->et.tr_imps = t->et.tr_imps * 0.4 + 0.6 * 1e6 * (double) imgs / (getus() - time_init_epic);

		}
		EVENT(t->OnfinishEpic, t);
	}
	t->process_id = PROCESS_ID_END;
}


void fitnes(Manage *t) {
	CHECK_REAL_TIME(t->real_time);

	REAL local_mse = 0;
	Tensor input;
	Tensor output;
	char label;
	int cnnLabel;
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
	for (; t->can_run && !t->cnn->erro->error && t->image < t->n_images2fitness; t->image++) {
		localimage = t->image + t->n_images2train;
		time_init_train = getus();
		input = t->imagens[localimage];
		output = t->targets[localimage];
		label = ((char *) t->labels->data)[localimage];
		t->cnn->predict(t->cnn, input);
		cnnLabel = t->cnn->maxIndex(t->cnn);
		local_mse = t->cnn->mseT(t->cnn, output);


		t->et.ft_info[label * t->et.ft_info_coluns + CASOS]++;
		if (cnnLabel == label) {
			t->et.ft_info[label * t->et.ft_info_coluns + HIT_RATE]++;
		} else {
			t->et.ft_info[label * t->et.ft_info_coluns + 3 + cnnLabel]++;
		}
		t->et.ft_info[label * t->et.ft_info_coluns + MSE] += local_mse;

		t->et.ft_imagem_atual = t->image;

		internal_time += getus() - time_init_train;
		t->current_time = internal_time;
		t->et.ft_time = t->current_time;
	}
//	EVENT(t->OnfinishFitnes, t);
	t->process_id = PROCESS_ID_END;
}

void loadImage(Manage *t) {
	if (t->cnn->erro->error)return;
	// obtem o tamanho de cada imagem
	size_t pixelsByImage = t->cnn->size_in.x * t->cnn->size_in.y * t->cnn->size_in.z;

	FILE *fimage = fopen(t->file_images, "rb");
	if (!fimage) {
		t->cnn->erro->error = 3;
		fprintf(stderr, "Imagens nao foram encontradas em %s\n", t->file_images);
		return;
	}
	Tensor tmp = Tensor_new(t->cnn->size_in.x , t->cnn->size_in.y , t->cnn->size_in.z, t->n_images, t->cnn->erro, TENSOR_CHAR | TENSOR_RAM | TENSOR4D, t->cnn->gpu->context, t->cnn->queue);

	// bytes de cabeçalho
	fread(tmp->data, 1, t->headers_images, fimage);
	// le as imagens;
	fread(tmp->data, pixelsByImage, t->n_images, fimage);
	fclose(fimage);
	Tensor imageInt = Tensor_new(tmp->x, tmp->y, tmp->z, tmp->w, t->cnn->erro, TENSOR_CHAR | TENSOR4D | TENSOR_CPY, t->cnn->gpu->context, t->cnn->queue, tmp->data);
	Release(tmp);
	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor imageREAL = Tensor_new(imageInt->x, imageInt->y, imageInt->z, imageInt->w, t->cnn->erro, TENSOR4D, t->cnn->gpu->context, t->cnn->queue);


	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao criar tensores imageREAL: ");
		Release(imageInt);
		return;
	}
	t->cnn->normalizeIMAGE(t->cnn, imageREAL, imageInt);

	t->cnn->erro->error = clFinish(t->cnn->queue);


	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao iniciar Kernel: ");
		Release(imageInt);
		Release(imageREAL);
		return;
	}

	Release(imageInt);


	for (int i = 0; i < t->n_images && t->can_run && !t->cnn->erro->error; ++i) {
		t->imagens[i] = Tensor_new(imageREAL->x, imageREAL->y, imageREAL->z, 1, t->cnn->erro, t->use_gpu_mem ? 0 : TENSOR_RAM, t->cnn->gpu->context, t->cnn->queue);
		t->et.ld_imagem_atual = i;
		t->imagens[i]->copyM(t->imagens[i], imageREAL, 0, i * t->imagens[i]->bytes, t->imagens[i]->bytes);
	}
	Release(imageREAL);
	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha enquanto roda o kernel: ");
	}
}

void loadLabels(Manage *t) {
	if (t->cnn->erro->error)return;
	FILE *flabel = fopen(t->file_labels, "rb");
	t->et.ll_imagem_atual = -1;
	if (!flabel) {
		t->cnn->erro->error = 3;
		fprintf(stderr, "Imagens nao foram encontradas em %s\n", t->file_labels);
		return;
	}
	// bytes de cabeçalho
	fread(t->labels->data, 1, t->headers_labels, flabel);
	// le as imagens;
	fread(t->labels->data, 1, t->n_images, flabel);
	fclose(flabel);
	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor labelInt = Tensor_new(1, t->n_images, 1, 1, t->cnn->erro, TENSOR_CHAR | TENSOR_CPY, t->cnn->gpu->context, t->cnn->queue, t->labels->data);
	Tensor labelREAL = Tensor_new(1, t->n_classes, 1, t->n_images, t->cnn->erro, TENSOR4D, t->cnn->gpu->context, t->cnn->queue, t->labels->data);

	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao criar tensores imageREAL: ");
		return;
	}
	t->cnn->extractVectorLabelClass(t->cnn, labelREAL, labelInt);
	Release(labelInt);
	if (t->cnn->erro->error) {
		fprintf(stderr, "Falha ao iniciar kernel: ");
		Release(labelREAL);
		return;
	}
	for (int i = 0; i < t->n_images && t->can_run; ++i) {
		t->targets[i] = Tensor_new(1, t->n_classes, 1, 1, t->cnn->erro, 0, t->cnn->gpu->context, t->cnn->queue);
		t->targets[i]->copyM(t->targets[i], labelREAL, 0, i * t->targets[i]->bytes, t->targets[i]->bytes);
		t->et.ll_imagem_atual = i;
	}
	Release(labelREAL);
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

void Manage_release(Manage *t) {
	t->can_run = 0;
	releaseProcess(&t->process);
	Release(t->cnn);
	if (t->imagens) {
		for (int i = 0; i < t->n_images; i++) {
			Release(t->imagens[i]);
		}
		free_mem(t->imagens);
	}
	ReleaseStr(t->file_labels);
	ReleaseStr(t->file_images);
	ReleaseStr(t->homePath);
	ReleaseStr(t->treino_info);
	ReleaseStr(t->fitnes_info);
	ReleaseStr(t->class_names);
	if (t->targets) {
		for (int i = 0; i < t->n_images; i++) {
			Release(t->targets[i]);
		}
		free_mem(t->targets);
	}


	Release(t->labels);
	releaseEstatitica(&t->et);
	releaseStringsManageTrain(t);
	*t = (Manage) {0};
	if (t->self_release)
		free_mem(t);

}


void releaseStringsManageTrain(Manage *t) {
	if (t->file_images)free_mem(t->file_images);
	if (t->file_labels)free_mem(t->file_labels);
	if (t->class_names)free_mem(t->class_names);
	if (t->homePath)free_mem(t->homePath);
}

void Manage_to_workDir(Manage *t) {

	if (!t->cnn || t->cnn->erro->error)return;

	if (SetDir(t->homePath)) {
		t->cnn->erro->error = 4;
		return;
	};
}

void Manage_setEvent(ManageEvent *dst, ManageEvent src) {
	*dst = src;
}

void Manage_run(Manage *t, int run) {
	t->can_run = run != 0;
}



Manage Manage_new(char *luafile, int luaIsProgram) {

	Manage result = {0};
	result.cnn = Cnn_new();

	if (!luaIsProgram)
		CnnLuaLoadFile(result.cnn, luafile);
	else
		CnnLuaLoadString(result.cnn, luafile);

	Manage_update_lua_params(&result);
	result.can_run = 1;
	Manage_to_workDir(&result);
	return result;
}

int Manage_load(Manage *t, int runBackground) {
	t->real_time = 1;
	t->process_id = PROCESS_ID_LOAD;
	releaseProcess(&t->process);
	if (runBackground) {
		t->process = Thread_new(loadData, t);
		return t->process != NULL;
	}
	loadData(t);
	return 0;
}

int Manage_train(Manage *t, int runBackground) {

	t->real_time = 1;
	t->process_id = PROCESS_ID_TRAIN;
	releaseProcess(&t->process);

	if (runBackground) {
		t->process = Thread_new(train, t);
		return t->process != NULL;
	}
	train(t);
	return 0;
}

int Manage_fitnes(Manage *t, int runBackground) {
	t->real_time = 1;
	t->process_id = PROCESS_ID_FITNES;
	releaseProcess(&t->process);
	if (runBackground) {
		t->process = Thread_new(fitnes, t);
		return t->process != NULL;
	}
	fitnes(t);
	return 0;
}

void waitEndProces(Manage *t) {
	while (t->process_id != PROCESS_ID_END) {
		Sleep(1);
		printf("wainting %d\n", t->process_id);
	}
}

void __manageTrainloop__(Manage *t) {
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
	while (t->process_id != PROCESS_ID_END) {
		EVENT(ev, t);
		Sleep(1);
	}
	EVENT(ev, t);
}

void Manage_loop(Manage *t, int run_background) {

	if (run_background) {
		if (t->update_loop) {
			ThreadKill(t->update_loop, -1);
			ThreadClose(t->update_loop);
		}
		t->update_loop = Thread_new(__manageTrainloop__, t);
		return;
	}
	__manageTrainloop__(t);
}

#if (MANAGE_DEBUG_LOAD_LUA == 1)
#define manage_show_load_lua(format,...)printf(format,##__VA_ARGS__)
#else
#define manage_show_load_lua
#endif
#define MANAGE_GETLUA_INT(cvar, name)lua_getglobal(L,name);   \
if(!lua_isnoneornil(L,-1)&&luaL_checkinteger(L,-1))    (cvar) = lua_tonumber(L,-1); \
else fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
manage_show_load_lua("%s %d\n",#cvar,cvar);

#define MANAGE_GETLUA_BOLL(cvar, name)lua_getglobal(L,name);   \
if(!lua_isnoneornil(L,-1)&&lua_isboolean(L,-1)) (cvar) = lua_toboolean(L,-1); \
else fprintf(stderr,"warning: %s não instanciado em lua\n",name);\
manage_show_load_lua("%s %d\n",#cvar,cvar);

#define MANAGE_GETLUA_STR(cvar, name)lua_getglobal(L,name);    if(luaL_checkstring(L,-1)){                                 \
ReleaseStr(cvar);                                                                                                                           \
const char *__tmp__ = lua_tostring(L,-1);                                                                                        \
size_t len  =  strlen(__tmp__);                                                                                            \
 (cvar) = alloc_mem(len+1,1);                                                                                              \
 memcpy(cvar,__tmp__,len);\
} else fprintf(stderr,"warning: %s não instanciado em lua\n",name);                                                        \
manage_show_load_lua("%s %s\n",#cvar,cvar);

void Manage_update_lua_params(Manage *manage) {
	if (!manage->cnn)return;
	if (manage->cnn->erro->error)return;
	lua_State *L = manage->cnn->LuaVm;
	if (!L)return;
	MANAGE_GETLUA_INT(manage->n_epics, "Numero_epocas");
	MANAGE_GETLUA_STR(manage->homePath, "home");
	MANAGE_GETLUA_STR(manage->name, "nome");
	MANAGE_GETLUA_INT(manage->n_images, "Numero_Imagens");
	MANAGE_GETLUA_INT(manage->n_images2train, "Numero_ImagensTreino");
	MANAGE_GETLUA_INT(manage->n_images2fitness, "Numero_ImagensAvaliacao");
	MANAGE_GETLUA_INT(manage->n_classes, "Numero_Classes");
	MANAGE_GETLUA_STR(manage->class_names, "classes");
	MANAGE_GETLUA_INT(manage->headers_images, "bytes_remanessentes_imagem");
	MANAGE_GETLUA_BOLL(manage->use_gpu_mem, "gpu_mem");
	MANAGE_GETLUA_INT(manage->headers_labels, "bytes_remanessentes_classes");
	MANAGE_GETLUA_STR(manage->treino_info, "estatisticasDeTreino");
	MANAGE_GETLUA_STR(manage->fitnes_info, "estatiscasDeAvaliacao");
	MANAGE_GETLUA_STR(manage->file_images, "arquivoContendoImagens");
	MANAGE_GETLUA_STR(manage->file_labels, "arquivoContendoRespostas");
}
