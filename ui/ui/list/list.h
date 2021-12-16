//
// Created by Henrique on 15/12/2021.
//

#ifndef UI_LIST_H
#define UI_LIST_H

#include <stdint.h>

typedef struct {
	 int length;
	uint8_t _lock_;
	void **elements;
	void (*push)(void *self, void *data);
	void *(*pop)(void *self);
	void (*lock)(void *self);
	void (*unlock)(void *self);
} Lista;

Lista Lista_new();
#endif //UI_LIST_H
