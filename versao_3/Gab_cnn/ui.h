//
// Created by hslhe on 10/11/2021.
//

#ifndef CNN_GPU_UI_H
#define CNN_GPU_UI_H

#include "conio2/conio2.h"

typedef struct {
	int x0, y0;
	char text[250];
	double value;
	int init;
	int end;
} Progress;
atomic_int blockTHread = 0;

char *asprintf(size_t *tlen, const char *format, ...) {
	size_t len = 0;
	char *tmp = NULL;
	va_list v;
	va_start(v, format);
	len = vsnprintf(NULL, 0, format, v) + 1;
	tmp = alloc_mem(len, 1);
	vsnprintf(tmp, len, format, v) + 1;
	va_end(v);
	if (tlen)
		*tlen = len;
	return tmp;
}

void showP(Progress *p, double value) {
	if (value > 1)return;
	if (p->init && (p->end || p->value == value))return;
	p->value = value;
	while (blockTHread);
	p->
			init = 1;
	blockTHread = 1;
	gotoxy(p->x0, p->y0);
	if (value == 1)
		p->end = 1;
	value = value * 100;
	size_t tsize;
	char *tmp = asprintf(&tsize, "%s % 3d.%02d%%", p->text, (int) value, (int) (value * 100 - ((int) value) * 100));
	int green = tsize * value / 100;
	struct text_info info;
	gettextinfo(&info);
	textbackground(GREEN);
	int i = 0;
	for (i = 0; i < green && tmp[i]; ++i) {
		printf("%c", tmp[i]);
	}
	textattr(info.attribute);
	for (; tmp[i]; ++i) {
		printf("%c", tmp[i]);
	}
	free_mem(tmp);

	blockTHread = 0;
}

Progress newProgress(char *text, double value, int x0, int y0, int size) {
	for (int i = 0; text[i]; ++i) {
		if (text[i] == '\n' || text[i] == '\r') {
			text[i] = ' ';
		}
	}
	Progress p = {0};
	p.x0 = x0;
	p.y0 = y0;
	p.value = value;
	size = size > 240 ? 240 : size;
	strncpy(p.text, text, size);
	int i = strlen(p.text);
	for (; i < size; ++i) {
		p.text[i] = ' ';
	}
	return p;
}

#endif //CNN_GPU_UI_H
