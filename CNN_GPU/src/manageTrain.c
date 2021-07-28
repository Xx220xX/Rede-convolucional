//
// Created by Henrique on 28-Jul-21.
//

#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include"utils/time_utils.h"

#define ERROR(format, ...)    getClErrorWithContext(t->cnn->error.error, \
t->cnn->error.msg,EXCEPTION_MAX_MSG_SIZE,format, ## __VA_ARGS__)

#define EVENT(event, param) if(event)event(param)

void loadImage(ManageTrain *t);

void loadLabels(ManageTrain *t);

void loadImages(ManageTrain *t) {
	if (!t->cnn || t->cnn->error.error)return;
	Tensor entrada = t->cnn->camadas[0]->entrada;
	t->imagens = new_Tensor(t->cnn->cl->context, t->cnn->queue,
	                        TENSOR_HOST | TENSOR_SMEM | TENSOR4D,
	                        entrada->x, entrada->y, entrada->z, t->n_images,
	                        &t->cnn->error, NULL);
	t->labels = new_Tensor(t->cnn->cl->context, t->cnn->queue,
	                       TENSOR_HOST | TENSOR_SMEM | TENSOR_CHAR,
	                       t->n_images, 1, 1, 1,
	                       &t->cnn->error, NULL);
	t->targets = new_Tensor(t->cnn->cl->context, t->cnn->queue,
	                        TENSOR_HOST | TENSOR_SMEM,
	                        t->n_images, t->n_classes, 1, 1,
	                        &t->cnn->error, NULL);
	loadImage(t);
	loadLabels(t);
	EVENT(t->OnloadedImages, t);
}

void train(ManageTrain *t) {
	EVENT(t->OnInitTrain,t);

	double time_init_train;
	double internal_time = t->current_time;
	double *input;
	double *output;
	char label;
	char cnn_label;
	if (!t->et.acertos) {
		t->et.acertos = alloc_mem(t->n_images2train, sizeof(double));
		t->et.erros = alloc_mem(t->n_images2train, sizeof(double));
	}


	for (; t->can_run && !t->cnn->error.error && t->epic < t->n_epics; t->epic++) {
		if (t->image >= t->n_images) {
			t->image = 0;
			t->et.n = 0;
			t->et.max_size = t->n_images;
		}
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
			t->et.acertos[t->image] = t->sum_acerto / (t->image + 1.0);
			t->et.erros[t->image] = t->sum_erro / (t->image + 1.0);
			t->et.n = t->image;
			internal_time += getms() - time_init_train;
			t->current_time = internal_time;
		}
		EVENT(t->OnfinishEpic, t);
	}
	EVENT(t->OnfinishTrain, t);
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
	fread(t->imagens, 1, t->headers_images, fimage);
	// le as imagens;
	fread(t->imagens, pixelsByImage, t->n_images, fimage);
	fclose(fimage);

	Tensor imageInt = new_Tensor(t->cnn->cl->context, t->cnn->queue,
	                             TENSOR_CHAR | TENSOR4D,
	                             t->imagens->x, t->imagens->y, t->imagens->z, t->imagens->w,
	                             &t->cnn->error, NULL
	);
	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageInt: ");
		return;
	}
	Tensor imageDouble = new_Tensor(t->cnn->cl->context, t->cnn->queue,
	                                TENSOR_UPTR | TENSOR4D,
	                                t->imagens->x, t->imagens->y, t->imagens->z, t->imagens->w,
	                                &t->cnn->error, t->imagens->host);

	if (t->cnn->error.error) {
		ERROR("Falha ao criar tensores imageDouble: ");
		return;
	}
	t->cnn->error.error = TensorPutValuesMem(t->cnn->queue, imageInt, t->imagens->host, imageInt->bytes * imageInt->w);
	if (t->cnn->error.error) {
		ERROR("Falha ao recuperar dados do tensor: ");
		releaseTensor(&imageInt);
		releaseTensor(&imageDouble);
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
		ERROR("Falha ao iniciar kernel: ");
		releaseTensor(&imageInt);
		return;
	}

	t->cnn->error.error = synchronizeKernel(t->cnn->queue);
	if (t->cnn->error.error) {
		ERROR("Falha enquanto roda o kernel: ");
		releaseTensor(&imageInt);
		releaseTensor(&imageDouble);
		return;
	}
	releaseTensor(&imageInt);
	releaseTensor(&imageDouble);

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
	fread(t->labels, 1, t->headers_labels, flabel);
	// le as imagens;
	fread(t->imagens, 1, t->n_images, flabel);
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
	                                TENSOR_UPTR,
	                                t->targets->x, t->targets->y, 1, 1,
	                                &t->cnn->error, t->targets->host);

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

	t->cnn->error.error = synchronizeKernel(t->cnn->queue);
	if (t->cnn->error.error) {
		ERROR("Falha enquanto roda o kernel: ");
		releaseTensor(&labelInt);
		releaseTensor(&labelDouble);
		return;
	}
	releaseTensor(&labelInt);
	releaseTensor(&labelDouble);
}