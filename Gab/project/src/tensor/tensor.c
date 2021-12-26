//
// Created by hslhe on 11/11/2021.
//

#include "tensor/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>
#include "gpu/Gpu.h"
#include "error_list.h"

#define asprintf(str, format, ...){int len = snprintf(NULL,0,format,## __VA_ARGS__);\
str = calloc(len+1,1);                                                           \
snprintf(str,len,format,## __VA_ARGS__);}


char *Tensor_putvaluesAsstr(Tensor self) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return NULL; }

	Memory m = {0};
	m.mem = self->getvalues(self, NULL);
	size_t len = 0;
	char *string = NULL;
	for (int w = 0; w < self->w; ++w) {
		if (self->flag.dimensao4D) {
			if (w == 0) apendstr(string, len, "[")
			else apendstr(string, len, "\n,[")
		}

		for (int z = 0; z < self->z; ++z) {
			if (z == 0) apendstr(string, len, "[")
			else apendstr(string, len, ",[")

			for (int x = 0; x < self->x; ++x) {
				if (x == 0) apendstr(string, len, "[")
				else apendstr(string, len, ",[")
				for (int y = 0; y < self->y; ++y) {
					if (y > 0) apendstr(string, len, ", ");
					if (self->flag.inteiro) {
						apendstr(string, len, "%d", m.inteiro[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else if (self->flag.caractere) {
//						apendstr(string, len, "\"0x%02X\"", (int) m.caractere[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
						apendstr(string, len, "%d", (int) m.caractere[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else {
						apendstr(string, len, "%g", (double) m.real[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					}
				}
				apendstr(string, len, "]")
			}
			apendstr(string, len, "]")
		}

		if (self->flag.dimensao4D) {
			apendstr(string, len, "]")
		}
	}
	free(m.mem);
	ECXPOP(self->erro);
	return string;
}

char *Tensor_printhex(Tensor self) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return NULL; }
	size_t id;
	Memory m = {0};
	m.mem = self->getvalues(self, NULL);
	size_t len = 0;
	char *string = NULL;
	for (int w = 0; w < self->w; ++w) {
		if (self->flag.dimensao4D) {
			if (w == 0) apendstr(string, len, "[")
			else apendstr(string, len, "\n,[")
		}

		for (int z = 0; z < self->z; ++z) {
			if (z == 0) apendstr(string, len, "[")
			else apendstr(string, len, ",[")

			for (int x = 0; x < self->x; ++x) {
				if (x == 0) apendstr(string, len, "[")
				else apendstr(string, len, ",[")
				for (int y = 0; y < self->y; ++y) {
					if (y > 0) apendstr(string, len, ", ");
					id = y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y;
					id = id * self->size_element;
					apendstr(string, len, "\"0x%02X\"", (int) m.caractere[id])
					for (int i = 1; i < self->size_element; ++i) {
						apendstr(string, len, "%02X", (int) m.caractere[id + i])
					}
				}
				apendstr(string, len, "]")
			}
			apendstr(string, len, "]")
		}

		if (self->flag.dimensao4D) {
			apendstr(string, len, "]")
		}
	}
	free(m.mem);
	ECXPOP(self->erro);
	return string;
}

char *Tensor_json(Tensor self, int showValues) {
	ECXPUSH(self->erro);
	char *string;
	char *tmp = "";
	if (showValues == 2) {
		tmp = Tensor_printhex(self);
	} else if (showValues) {
		tmp = Tensor_putvaluesAsstr(self);
	}

	asprintf(string, "{\n"
			PAD"\"flag\":{\n"
			PAD PAD "\"dimensao4D\":%d,\n"
			PAD PAD"\"ram\":%d,\n"
			PAD PAD"\"caractere\":%d,\n"
			PAD PAD"\"inteiro\":%d,\n"
			PAD PAD"\"shared\":%d,\n"
			PAD PAD"\"size\":%llu\n"
			PAD"},\n"
			PAD"\"flagn\":%d,\n"
			PAD"\"x\":%d,\n"
			PAD"\"y\":%d,\n"
			PAD"\"z\":%d,\n"
			PAD"\"w\":%d,\n"
			PAD"\"length\":%d,\n"
			PAD"\"bytes\":%d,\n"
			PAD"\"size_element\":%d,\n"
			PAD"\"data\":[%s],\n"
			PAD"\"size\":%d\n"

			   "}", (int) self->flag.dimensao4D, (int) self->flag.ram, (int) self->flag.caractere, (int) self->flag.inteiro, (int) self->flag.shared, sizeof(TensorFlag), (int) self->flag.flag, self->x, self->y, self->z, self->w, self->length, self->bytes, self->size_element, tmp, sizeof(Tensor_t)

			)
	if (showValues) {
		gab_free(tmp);
	}
	ECXPOP(self->erro);
	return string;
}

void Tensor_release(Tensor *self) {
	if (!self) { return; }
	if (!*self) { return; }
	if ((*self)->data) {
		if ((*self)->flag.ram) {
			free((*self)->data);
		} else if ((*self)->flag.shared) {

		} else {
			clReleaseMemObject((*self)->data);
		}
	}
	gab_free(self[0]);
	*self = NULL;
}

void Tensor_registreError(Tensor self, char *format, ...) {
	va_list v;
	va_start(v, format);
	FILE *f;
	if (self->file_debug) {
		f = fopen(self->file_debug, "a");
	} else {
		f = stderr;
	}// fopen("tensor.debug", "a");
	vfprintf(f, format, v);
	va_end(v);
	if (self->file_debug) {
		fclose(f);
	}

}

int Tensor_setvalues(Tensor self, void *data) {
	return self->setvaluesM(self, 0, data, self->bytes);
}

void *Tensor_getvalues(Tensor self, void *data) {
	return self->getvaluesM(self, 0, data, self->bytes);
}

int Tensor_setvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return self->erro->error; }
	if (self->flag.ram) {
		memcpy(self->data + offset, data, bytes);
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");
	} else {
		clFlush(self->queue);
		clFinish(self->queue);
		self->erro->error = clEnqueueWriteBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	ECXPOP(self->erro);
	return self->erro->error;
}


void *Tensor_getvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return NULL; }
	if (!data) {
		data = calloc(bytes, 1);
	}
	if (self->flag.ram) {
		memcpy(data, self->data + offset, bytes);
	} else if (self->flag.shared) {
		fprintf(stderr, "ERROR: shared memory not implanted\n");

	} else {
		clFlush(self->queue);
		clFinish(self->queue);
		self->erro->error = clEnqueueReadBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	ECXPOP(self->erro);
	return data;
}

int Tensor_randomize(Tensor self, int type, REAL a, REAL b) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return self->erro->error; }
	void *m = calloc(self->bytes, 1);
	size_t len = self->bytes / self->size_element;
	REAL x;
	for (int i = 0; i < len; ++i) {
		if (type == TENSOR_GAUSSIAN) {
			x = Tensor_randn() * a + b;
		} else {
			x = Tensor_rand() * a + b;
		}
		if (self->flag.caractere) {
			((char *) m)[i] = (char) x;
		} else if (self->flag.inteiro) {
			((int *) m)[i] = (int) x;
		} else {
			((REAL *) m)[i] = (REAL) x;
		}
	}
	self->setvalues(self, m);
	free(m);
	ECXPOP(self->erro);
	return self->erro->error;
}


void normalizeReal(REAL *data, size_t len, REAL a) {
	REAL mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) { mx = data[i]; }
		if (mn > data[i]) { mn = data[i]; }
	}
//	printf("%g %g\n", mx, mn);
	mx = mx - mn;
	a = a / mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (data[i] - mn) * a;
	}
}

void normalizeInt(int *data, size_t len, REAL a) {

	int mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) { mx = data[i]; }
		if (mn > data[i]) { mn = data[i]; }
	}
	mx = mx - mn;
	a = a / (REAL) mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (int) ((REAL) (data[i] - mn) * a);
	}
}

int Tensor_imagegrayREAL(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	ECXPUSH(self->erro);
	if (self->erro->error) { return self->erro->error; }
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	REAL *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	normalizeReal(data, self->x * self->y, 255);
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	end:
	free(data);
	ECXPOP(self->erro);
	return self->erro->error;
}


int Tensor_imagegrayINT(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->erro->error) { return self->erro->error; }
	ECXPUSH(self->erro);
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	int *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	normalizeInt(data, self->x * self->y, 255);
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y] & 0xff);
		}
	}
	end:
	free(data);
	ECXPOP(self->erro);
	return self->erro->error;
}

int Tensor_imagegrayCHAR(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	if (self->erro->error) { return self->erro->error; }
	ECXPUSH(self->erro);
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	ubyte *data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	end:
	free(data);
	ECXPOP(self->erro);
	return self->erro->error;
}

int Tensor_fillM(Tensor self, size_t offset, size_t bytes, void *patern, size_t size_patern) {
	ECXPUSH(self->erro);
	if (!patern || size_patern <= 0) {
		self->erro->error = !patern ? GAB_NULL_POINTER_ERROR : GAB_INVALID_PARAM;
		return self->erro->error;
	}
	if (self->flag.ram) {
		void *p = self->data + offset;
		void *pend = self->data + offset + bytes;
		for (; p <= pend; p += size_patern) {
			if (p + size_patern <= pend) {
				memcpy(p, patern, size_patern);
			}
		}
	} else if (self->flag.shared) {

	} else {
		self->erro->error = clEnqueueFillBuffer(self->queue, self->data, patern, size_patern, offset, bytes, 0, NULL, NULL);
	}
	ECXPOP(self->erro);
	return self->erro->error;
}

int Tensor_fill(Tensor self, char partern) {
	Tensor_fillM(self, 0, self->bytes, &partern, 1);
}

int Tensor_copy(Tensor self, Tensor b) {
	ECXPUSH(self->erro);
	if (self->bytes != b->bytes) {
		fprintf(stderr, "O tensor b deve possuir o mesmo tamanho\n");
		self->erro->error = GAB_INDEX_OUT_OF_BOUNDS;
		return self->erro->error;
	}
	void *mem = NULL;
	if (b->flag.ram) {
		self->setvalues(self, b->data);
	} else if (self->flag.ram) {
		mem = b->getvalues(b, NULL);
		self->setvalues(self, mem);
		gab_free(mem);
	} else {
		self->erro->error = clEnqueueCopyBuffer(self->queue, b->data, self->data, 0, 0, self->bytes, 0, NULL, NULL);
		clFinish(self->queue);
	}
	ECXPOP(self->erro);
	return self->erro->error;
}

int Tensor_copyM(Tensor self, Tensor b, size_t self_ofset, size_t b_ofset, size_t bytes) {
	ECXPUSH(self->erro);
	void *mem = NULL;
	if (b->flag.ram) {
		self->setvaluesM(self, self_ofset, b->data + b_ofset, bytes);
	} else if (self->flag.ram) {
		mem = b->getvaluesM(b, b_ofset, NULL, bytes);
		self->setvaluesM(self, self_ofset, mem, bytes);
		gab_free(mem);
	} else {
		self->erro->error = clEnqueueCopyBuffer(self->queue, b->data, self->data, b_ofset, self_ofset, bytes, 0, NULL, NULL);
//		clFlush(self->queue);
		self->erro->setError(self->erro, clFinish(self->queue));
	}
	ECXPOP(self->erro);
	return self->erro->error;
}

void Tensor_fprint(Tensor self,FILE *f) {
	ECXPUSH(self->erro);
	char *json = Tensor_putvaluesAsstr(self);
	fprintf(f,"%u : %s\n", self->size_element, json);
	gab_free(json);
	ECXPOP(self->erro);
}
void Tensor_print(Tensor self) {
	ECXPUSH(self->erro);
	Tensor_fprint(self,stdout);
	ECXPOP(self->erro);
}

void Tensor_tomatlab(Tensor self, FILE *f, char *name, char *reshapeF) {
	ECXPUSH(self->erro);
	Memory memory;
	memory.mem = self->getvalues(self, NULL);
	if (!memory.mem) { return; }
	fprintf(f, "%s = [", name);
	for (int i = 0; i < self->length; ++i) {
		if (i > 0) { fprintf(f, ", "); }
		if (self->flag.caractere) {
			fprintf(f, "%d", (int) memory.caractere[i]);
		} else if (self->flag.inteiro) {
			fprintf(f, "%d", memory.inteiro[i]);
		} else {
			if (isnan(memory.real[i])) {
				fprintf(f, "nan");
			} else if (isinf(memory.real[i])) {
				fprintf(f, "%sinf", memory.real[i] > 0 ? "+" : "-");
			} else {
				fprintf(f, "%.16lf", (double) memory.real[i]);
			}
		}
	}
	fprintf(f, "];\n");
	if (reshapeF) {
		if (self->flag.dimensao4D) {
			fprintf(f, "%s = %s(%s,[%zu,%zu,%zu,%zu]);\n", name, reshapeF, name, self->x, self->y, self->z, self->w);
		} else {
			fprintf(f, "%s = %s(%s,[%zu,%zu,%zu]);\n", name, reshapeF, name, self->x, self->y, self->z);
		}
	}
	gab_free(memory.mem);
	ECXPOP(self->erro);
}

#include "png/png.h"

void Tensor_png(Tensor self, const char *file) {
	ECXPUSH(self->erro);
	ubyte *img = gab_alloc(self->length, 1);
	size_t largura = self->y * self->z;
	size_t altura = self->x * self->w;
	for (int w = 0; w < self->w; ++w) {
		for (int z = 0; z < self->z; ++z) {
			self->imagegray(self, img, largura, altura, self->y, self->x, w * self->x, z * self->y, z, w);
		}
	}

	pngGRAY(file, img, largura, altura);

	gab_free(img);
	ECXPOP(self->erro);
}

int Tensor_map(Tensor self, void (*fmap)(Tensor self, void *el, int i, int j, int z, int w, int k)) {
	ECXPUSH(self->erro);
	void *data = self->getvalues(self, NULL);
	int k;
	for (int w = 0; w < self->w; ++w) {
		for (int z = 0; z < self->z; ++z) {
			for (int i = 0; i < self->x; ++i) {
				for (int j = 0; j < self->y; ++j) {
					k = j + i * self->y + z * self->y * self->x + w * self->x * self->y * self->z;
					fmap(self, data + self->size_element * k, i, j, z, w, k);
				}
			}
		}
	}
	self->setvalues(self, data);
	gab_free(data);
	ECXPOP(self->erro);
	return self->erro->error;
}

REAL Tensor_media(Tensor self) {
	ECXPUSH(self->erro);
	REAL media = 0;
	void *dt = self->getvalues(self, NULL);
	if (self->flag.inteiro) {
		int *idt = dt;
		for (int i = 0; i < self->length; ++i) {
			media += idt[i];
		}
	} else if (self->flag.caractere) {
		unsigned char *cdt = dt;

		for (int i = 0; i < self->length; ++i) {
			media += cdt[i];
		}
	} else {
		REAL *rdt = dt;
		for (int i = 0; i < self->length; ++i) {
			media += rdt[i];
		}
	}
	gab_free(dt);
	ECXPOP(self->erro);
	return media / self->length;
}

REAL Tensor_var(Tensor self) {
	ECXPUSH(self->erro);
	REAL media = self->media(self);
	REAL var = 0;
	REAL tmp;
	void *dt = self->getvalues(self, NULL);
	if (self->flag.inteiro) {
		int *idt = dt;
		for (int i = 0; i < self->length; ++i) {
			tmp = (idt[i] - media);
			var += tmp * tmp;
		}
	} else if (self->flag.caractere) {
		unsigned char *cdt = dt;

		for (int i = 0; i < self->length; ++i) {
			tmp = (cdt[i] - media);
			var += tmp * tmp;
		}
	} else {
		REAL *rdt = dt;
		for (int i = 0; i < self->length; ++i) {
			tmp = (rdt[i] - media);
			var += tmp * tmp;
		}
	}
	gab_free(dt);
	ECXPOP(self->erro);
	return var / self->length;
}

REAL Tensor_std(Tensor self) {
	ECXPUSH(self->erro);
	REAL var = self->var(self);
	ECXPOP(self->erro);
	return sqrt(var);
}

/**
 * 8 bytes -> tamanho total do tensor
 * 8 bytes -> dimensão x
 * 8 bytes -> dimensão y
 * 8 bytes -> dimensão z
 * 8 bytes -> dimensão w
 * 1 byte -> tipo do tensor  double :0\n
 * 							 float: 1\n
 * 							 int: 2\n
 * 							 char: 3\n
 * n bytes -> dados do tensor
 * @param self
 * @return
 */
void *Tensor_serialize(Tensor self, size_t *length) {
	size_t len = self->bytes + 5 * sizeof(size_t) + 1;
	if (length) { *length = len; }
	char type = USEFLOAT;
	void *data = gab_alloc(len, 1);
	self->getvalues(self, data + (5 * sizeof(size_t) + 1));
	memcpy(data + 0 * sizeof(size_t), &self->length, sizeof(size_t));
	memcpy(data + 1 * sizeof(size_t), &self->x, sizeof(size_t));
	memcpy(data + 2 * sizeof(size_t), &self->y, sizeof(size_t));
	memcpy(data + 3 * sizeof(size_t), &self->z, sizeof(size_t));
	memcpy(data + 4 * sizeof(size_t), &self->w, sizeof(size_t));

	if (self->flag.inteiro) {
		type = 2;
	} else if (self->flag.caractere) {
		type = 3;
	}
	memcpy(data + 5 * sizeof(size_t), &type, 1);
	return data;
}

Tensor Tensor_new(size_t x, size_t y, size_t z, size_t w, Ecx ecx, int flag, ...) {
	ECXPUSH(ecx);
	Tensor self = calloc(1, sizeof(Tensor_t));
	self->erro = ecx;
	memset((void*)&self->flag, flag, 1);
	// verificando flag
	if (self->flag.inteiro && self->flag.caractere) {
		Tensor_registreError(self, "flag invalida: inteiro e caractere = 1");
		free(self);
		ecx->error = GAB_INVALID_PARAM;
		return NULL;
	}
	if (self->flag.ram && self->flag.shared) {
		Tensor_registreError(self, "flag invalida: ram e shared = 1");
		free(self);
		ecx->error = GAB_INVALID_PARAM;
		return NULL;
	}
	self->x = x;
	self->y = y;
	self->z = z;
	self->w = w;
	if (!self->flag.dimensao4D) {
		self->w = 1;
	}
	self->size_element = sizeof(REAL);
	if (self->flag.inteiro) {
		self->size_element = sizeof(int);
	} else if (self->flag.caractere) {
		self->size_element = sizeof(char);
	}
	self->length = self->x * self->y * self->z * self->w;
	self->bytes = self->length * self->size_element;

	// alocar memoria
	if (self->flag.ram) {
		self->data = calloc(self->x * self->y * self->z * self->w, self->size_element);
	} else if (self->flag.shared) {
		fprintf(stderr, "Invalid flag: Shared memory not suported");
		exit(GAB_INVALID_PARAM);
	} else {
		va_list v;
		va_start(v, flag);
		self->context = va_arg(v, void *);
		self->queue = va_arg(v, void *);
		void *p = NULL;

		if (self->flag.copy) {
			p = va_arg(v, void *);
			self->data = clCreateBuffer(self->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, self->bytes, p, &self->erro->error);
		} else {
			self->data = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->bytes, NULL, &self->erro->error);
		}
		va_end(v);

	}

	// metodos
	self->json = Tensor_json;
	self->release = Tensor_release;
	self->registreError = Tensor_registreError;
	self->copy = Tensor_copy;
	self->serialize = Tensor_serialize;
	self->map = Tensor_map;
	self->copyM = Tensor_copyM;
	self->print = Tensor_print;
	self->fprint = Tensor_fprint;

	self->valuesStr = Tensor_putvaluesAsstr;
	self->tomatlab = Tensor_tomatlab;
	self->setvalues = Tensor_setvalues;
	self->getvalues = Tensor_getvalues;
	self->getvaluesM = Tensor_getvaluesm;
	self->setvaluesM = Tensor_setvaluesm;
	self->randomize = Tensor_randomize;
	self->fill = Tensor_fill;
	self->fillM = Tensor_fillM;
	self->png = Tensor_png;
	self->media = Tensor_media;
	self->var = Tensor_var;
	self->std = Tensor_std;

	if (self->flag.inteiro) {
		self->imagegray = Tensor_imagegrayINT;
	} else if (self->flag.caractere) {
		self->imagegray = Tensor_imagegrayCHAR;
	} else {
		self->imagegray = Tensor_imagegrayREAL;
	}
	ECXPOP(ecx);
	return self;
}

