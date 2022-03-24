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
	ECX_RETURN_IF_ERROR(self->ecx, NULL)
	char *string = NULL;
	Memory m = {0};
	size_t len = 0;
	double tmp;
	m.mem = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
	for (int w = 0; w < self->w; ++w) {
		if (self->flag.dimensao4D) {
			if (w == 0) apendstr(string, len, "[") else apendstr(string, len, "\n, [")
		}

		for (int z = 0; z < self->z; ++z) {
			if (z == 0) apendstr(string, len, "[") else apendstr(string, len, "\n, [")

			for (int x = 0; x < self->x; ++x) {
				if (x == 0) apendstr(string, len, "[") else apendstr(string, len, "\n, [")
				for (int y = 0; y < self->y; ++y) {
					if (y > 0) apendstr(string, len, ", ");
					if (self->flag.inteiro) {
						apendstr(string, len, "%d", m.inteiro[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else if (self->flag.caractere) {
//						apendstr(string, len, "\"0x%02X\"", (int) m.caractere[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
						apendstr(string, len, "%d", (int) m.caractere[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y])
					} else {
						tmp = m.real[y + x * self->y + z * self->x * self->y + w * self->z * self->x * self->y];

						apendstr(string, len, "%s%.5f", tmp > 0 ? " " : "", tmp)
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
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return string;
}

char *Tensor_printhex(Tensor self) {
	ECX_RETURN_IF_ERROR(self->ecx, NULL)

	char *string = NULL;
	size_t id;
	size_t len = 0;
	Memory m = {0};
	m.mem = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
	for (int w = 0; w < self->w; ++w) {
		if (self->flag.dimensao4D) {
			if (w == 0) apendstr(string, len, "[") else apendstr(string, len, "\n,[")
		}

		for (int z = 0; z < self->z; ++z) {
			if (z == 0) apendstr(string, len, "[") else apendstr(string, len, ",[")

			for (int x = 0; x < self->x; ++x) {
				if (x == 0) apendstr(string, len, "[") else apendstr(string, len, ",[")
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
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return string;
}

char *Tensor_json(Tensor self, int showValues) {
	ECX_RETURN_IF_ERROR(self->ecx, NULL)
	char *string = NULL;
	char *tmp = "";
	if (showValues == 2) {
		tmp = Tensor_printhex(self);
	} else if (showValues) {
		tmp = Tensor_putvaluesAsstr(self);
	}
	ECX_IF_FAILED(self->ecx, end)
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
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return string;
}

void Tensor_release(Tensor *self) {
	if (!self) {
		return;
	}
	if (!*self) {
		return;
	}
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

void Tensor_registreError(Tensor self, char *format, ...)  __attribute__ ((deprecated));

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
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int r = self->setvaluesM(self, 0, data, self->bytes);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return r;
}

void *Tensor_getvalues(Tensor self, void *data) {
	ECX_RETURN_IF_ERROR(self->ecx, NULL)
	void *r = self->getvaluesM(self, 0, data, self->bytes);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return r;
}

int Tensor_setvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	if (self->flag.ram) {
		memcpy(self->data + offset, data, bytes);
	} else if (self->flag.shared) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "ERROR: shared memory not implanted\n");
		goto end;
	} else {
		clFlush(self->queue);
		clFinish(self->queue);
		self->ecx->error = clEnqueueWriteBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}


void *Tensor_getvaluesm(Tensor self, size_t offset, void *data, size_t bytes) {
	ECX_RETURN_IF_ERROR(self->ecx, NULL)
	if (!data) {
		data = calloc(bytes, 1);
	}
	if (self->flag.ram) {
		memcpy(data, self->data + offset, bytes);
	} else if (self->flag.shared) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "ERROR: shared memory not implanted\n");
		goto end;
		//fprintf(stderr, "ERROR: shared memory not implanted\n");

	} else {
		clFlush(self->queue);
		clFinish(self->queue);
		self->ecx->error = clEnqueueReadBuffer(self->queue, self->data, CL_TRUE, offset, bytes, data, 0, NULL, NULL);
	}
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return data;
}

int Tensor_randomize(Tensor self, int type, REAL a, REAL b) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	void *m;
	size_t len = self->bytes / self->size_element;

	m = calloc(self->bytes, 1);
	REAL x;
	for (int i = 0; i < len; ++i) {
		switch (type) {
			case TENSOR_UNIFORM:
				x = Tensor_rand() * a + b;
				break;
			case TENSOR_GAUSSIAN:
				x = Tensor_randn() * a + b;
				break;
			case TENSOR_RANDINT:
				x = Tensor_randi() % ((int) a) + b;
				break;
			case TENSOR_UNIFORM | TENSOR_UNITARIO:
				x = (Tensor_rand()) / (self->length);
				break;
			case TENSOR_GAUSSIAN | TENSOR_UNITARIO:
				x = (Tensor_rand() + b) / (self->length);
				break;
			default:
				fprintf(stderr, "Invalid param type = %d\n", type);
				self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "Invalid param type = %d\n", type);
				goto end;

		}
		if (type == TENSOR_GAUSSIAN) {
			x = Tensor_randn() * a + b;
		} else if (type == TENSOR_UNIFORM) {
			x = Tensor_rand() * a + b;
		} else if (type == TENSOR_RANDINT) {
//			x = Tensor_randi() % ((int)a)) + b;
			int tmp = Tensor_randi();
			x = tmp % (int) a;
			x = x + b;
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
	end:
	if (m) {
		free(m);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}


int normalizeReal(REAL *data, size_t len, REAL a) {
	REAL mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) {
			mx = data[i];
		}
		if (mn > data[i]) {
			mn = data[i];
		}
	}
//	printf("%g %g\n", mx, mn);
	mx = mx - mn;
	if (mx == 0 || isinf(mx) || isnan(mx)) {
		return GAB_INVALID_DIVISION;
	}
	a = a / mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (data[i] - mn) * a;
	}
	return 0;
}

int normalizeInt(int *data, size_t len, REAL a) {
	int mx = data[0], mn = data[0];
	for (int i = 1; i < len; ++i) {
		if (mx < data[i]) {
			mx = data[i];
		}
		if (mn > data[i]) {
			mn = data[i];
		}
	}
	mx = mx - mn;
	if (mx == 0) {
		return GAB_INVALID_DIVISION;
	}
	a = a / (REAL) mx;
	for (int i = 0; i < len; ++i) {
		data[i] = (int) ((REAL) (data[i] - mn) * a);
	}
	return 0;
}

int Tensor_imagegrayREAL(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	REAL *data = NULL;


	data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	self->ecx->setError(self->ecx, normalizeReal(data, self->x * self->y, 255), "Divisao por 0");
	ECX_IF_FAILED(self->ecx, end)
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
//			self->ecx->error = GAB_INDEX_OUT_OF_BOUNDS;
			self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	end:
	if (data) {
		free(data);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}


int Tensor_imagegrayINT(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	int *data = NULL;
	data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	ECX_TRY(self->ecx, normalizeInt(data, self->x * self->y, 255), end, GAB_INVALID_DIVISION, "divisao por zero");

	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y] & 0xff);
		}
	}
	end:
	if (data) {
		free(data);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

int Tensor_imagegrayCHAR(Tensor self, ubyte *image, size_t width, size_t height_tensor, size_t w, size_t h, size_t i0, size_t j0, size_t z, size_t l) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int x, y;
	double px = ((double) self->x / (double) h), py = ((double) self->y / (double) w);
	ubyte *data = NULL;
	data = self->getvaluesM(self, (z * self->x * self->y + l * self->z * self->x * self->y) * self->size_element, NULL, self->size_element * self->x * self->y);
	for (int i = 0; i < h; ++i) {
		if (i + i0 >= height_tensor) {
			self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
			goto end;
		}
		for (int j = 0; j < w; ++j) {
			y = j * py;
			x = i * px;
			if (j + j0 >= width) {
				self->ecx->setError(self->ecx, GAB_INDEX_OUT_OF_BOUNDS, "Violacao de memoria");
				goto end;
			}
			image[(i + i0) * width + j + j0] = (char) (data[x * self->y + y]);
		}
	}
	end:
	if (data) {
		free(data);
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

int Tensor_fillM(Tensor self, size_t offset, size_t bytes, void *patern, size_t size_patern) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	if (!patern) {
		self->ecx->setError(self->ecx, GAB_NULL_POINTER_ERROR, "Pattern nao pode ser nulo");
		goto end;
	}
	if (!size_patern) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "size_patern nao pode ser 0");
		goto end;
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
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "shared memore nao implementado");
		goto end;
	} else {
		self->ecx->error = clEnqueueFillBuffer(self->queue, self->data, patern, size_patern, offset, bytes, 0, NULL, NULL);
	}
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

int Tensor_fill(Tensor self, char partern) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	int r = Tensor_fillM(self, 0, self->bytes, &partern, 1);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return r;
}

int Tensor_copy(Tensor self, Tensor b) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	void *mem = NULL;
	if (self->bytes != b->bytes) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "O tensor b deve possuir o mesmo tamanho\n");
		goto end;
	}

	if (b->flag.ram) {
		self->setvalues(self, b->data);
	} else if (self->flag.ram) {
		mem = b->getvalues(b, NULL);
		self->setvalues(self, mem);
		gab_free(mem);
	} else {
		self->ecx->error = clEnqueueCopyBuffer(self->queue, b->data, self->data, 0, 0, self->bytes, 0, NULL, NULL);
		clFinish(self->queue);
	}
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

int Tensor_copyM(Tensor self, Tensor b, size_t self_ofset, size_t b_ofset, size_t bytes) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	void *mem = NULL;
	if (b->flag.ram) {
		self->setvaluesM(self, self_ofset, b->data + b_ofset, bytes);
	} else if (self->flag.ram) {
		mem = b->getvaluesM(b, b_ofset, NULL, bytes);
		self->setvaluesM(self, self_ofset, mem, bytes);
		gab_free(mem);
	} else {
		self->ecx->error = clEnqueueCopyBuffer(self->queue, b->data, self->data, b_ofset, self_ofset, bytes, 0, NULL, NULL);
//		clFlush(self->queue);
		self->ecx->setError(self->ecx, clFinish(self->queue), "Falha ao copiar tensor\n");
	}
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

void Tensor_fprint(Tensor self, FILE *f) {
	ECX_RETURN_IF_ERROR(self->ecx,)
	char *json = Tensor_putvaluesAsstr(self);
	fprintf(f, ":\n%s\n", json);
	gab_free(json);
	/*int len = 40;
	int ofset = 0;
	if (len >= self->length / 2) {
		len = self->length;
	} else {
		ofset = self->length / 2;
	}
	void *data = gab_alloc(len, self->size_element);
	self->getvaluesM(self, ofset * self->size_element, data, len * self->size_element);
	fprintf(f,"[..., ");
	for (int i = 0; i < len; ++i) {
		if (self->flag.caractere) {
			fprintf(f, "%d, ", (int)((unsigned char*) data)[i]);
		} else if (self->flag.inteiro) {
			fprintf(f, "%d, ", ((int *) data)[i]);
		} else {
			fprintf(f, "%.4f, ", ((REAL *) data)[i]);
		}
	}
	fprintf(f," ...]\n");
	gab_free(data);
*/
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

void Tensor_print(Tensor self) {
	ECX_RETURN_IF_ERROR(self->ecx,)
	Tensor_fprint(self, stdout);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

void Tensor_tomatlab(Tensor self, FILE *f, char *name, char *reshapeF) {
	ECX_RETURN_IF_ERROR(self->ecx,)

	Memory memory = {0};
	memory.mem = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
	fprintf(f, "%s = [", name);
	for (int i = 0; i < self->length; ++i) {
		if (i > 0) {
			fprintf(f, ", ");
		}
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
	end:
	gab_free(memory.mem);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

#include "png/png.h"

void Tensor_png(Tensor self, const char *file) {
	ECX_RETURN_IF_ERROR(self->ecx,)

	ubyte *img = gab_alloc(self->length, 1);
	size_t largura = self->y * self->z;
	size_t altura = self->x * self->w;
	for (int w = 0; w < self->w; ++w) {
		for (int z = 0; z < self->z; ++z) {
			self->imagegray(self, img, largura, altura, self->y, self->x, w * self->x, z * self->y, z, w);
			ECX_IF_FAILED(self->ecx, end)
		}

	}

	pngGRAY(file, img, largura, altura);
	end:
	gab_free(img);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
}

int Tensor_map(Tensor self, void (*fmap)(Tensor self, void *el, int i, int j, int z, int w, int k)) {
	ECX_RETURN_IF_ERROR(self->ecx, self->ecx->error)
	void *data = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
	int k;
	for (int w = 0; w < self->w; ++w) {
		for (int z = 0; z < self->z; ++z) {
			for (int i = 0; i < self->x; ++i) {
				for (int j = 0; j < self->y; ++j) {
					k = j + i * self->y + z * self->y * self->x + w * self->x * self->y * self->z;
					fmap(self, data + self->size_element * k, i, j, z, w, k);
					ECX_IF_FAILED(self->ecx, end)
				}
			}
		}
	}
	self->setvalues(self, data);
	end:
	gab_free(data);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self->ecx->error;
}

REAL Tensor_media(Tensor self) {
	ECX_RETURN_IF_ERROR(self->ecx, 0)
	REAL media = 0;
	void *dt = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
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
	if (self->length == 0) {
		self->ecx->setError(self->ecx, GAB_INVALID_DIVISION, "O tamanho do tensor é nulo");
		goto end;
	}
	media = media / self->length;
	end:
	gab_free(dt);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return media;
}

REAL Tensor_var(Tensor self) {
	ECX_RETURN_IF_ERROR(self->ecx, 0)
	REAL media = self->media(self);
	ECX_IF_FAILED(self->ecx, end)
	REAL var = 0;
	REAL tmp;
	void *dt = self->getvalues(self, NULL);
	ECX_IF_FAILED(self->ecx, end)
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
	if (self->length == 0) {
		self->ecx->setError(self->ecx, GAB_INVALID_DIVISION, "O tamanho do tensor é nulo");
		goto end;
	}
	var = var / self->length;
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return var;
}

REAL Tensor_std(Tensor self) {
	ECX_RETURN_IF_ERROR(self->ecx,0)
	REAL var = self->var(self);
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
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
	ECX_RETURN_IF_ERROR(self->ecx,0)
	size_t len = self->bytes + 5 * sizeof(size_t) + 1;
	if (length) {
		*length = len;
	}
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
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return data;
}

Tensor Tensor_new(size_t x, size_t y, size_t z, size_t w, Ecx ecx, int flag, ...) {
	ECX_RETURN_IF_ERROR(ecx,NULL)
	Tensor self = calloc(1, sizeof(Tensor_t));
	self->ecx = ecx;
	memset((void *) &self->flag, flag, 1);
	// verificando flag
	if (self->flag.inteiro && self->flag.caractere) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "flag invalida: inteiro e caractere = 1");
		free(self);
		self = NULL;
		goto end;
	}
	if (self->flag.ram && self->flag.shared) {
		self->ecx->setError(self->ecx, GAB_INVALID_PARAM, "flag invalida: ram e shared = 1");
		free(self);
		self = NULL;
		goto end;
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
		if (!self->data) {
			self->ecx->setError(self->ecx, GAB_INVALID_MEMORY, "Retorno nullo de calloc");
			gab_free(self);
			self = NULL;
			goto end;
		}
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
			self->data = clCreateBuffer(self->context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, self->bytes, p, &self->ecx->error);
		} else {
			self->data = clCreateBuffer(self->context, CL_MEM_READ_WRITE, self->bytes, NULL, &self->ecx->error);
		}
		if (self->ecx->error) {
			self->ecx->pushMsg(self->ecx, "Retorno nullo de clCreateBuffer");
			gab_free(self);
			self = NULL;
			goto end;
		}
		va_end(v);

	}

	// metodos
	self->json = Tensor_json;
	self->release = Tensor_release;
	self->registreError = NULL;
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
	end:
	ECX_REGISTRE_FUNCTION_IF_ERROR(self->ecx)
	return self;
}

