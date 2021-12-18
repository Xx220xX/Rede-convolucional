//
// Created by Henrique on 15/12/2021.
//

#include "list.h"
#include <stdlib.h>

void Lista_push(Lista *self, void *element) {
	self->lock(self);
	self->elements = realloc(self->elements, (self->length+1) * sizeof(void *));
	self->elements[self->length] = element;
	self->length++;
	self->unlock(self);
}

void *Lista_pop(Lista *self) {
	self->lock(self);
	if (self->length <= 0) { return NULL; }
	self->length--;
	void *value = self->elements[self->length];
	self->elements = realloc(self->elements, self->length * sizeof(void *));
	self->unlock(self);
	return value;

}

void Lista_lock(Lista *self) {
	while (self->_lock_);
	self->_lock_ = 1;
}

void Lista_unlock(Lista *self) {
	self->_lock_ = 0;
}

Lista Lista_new() {
	return (Lista) {._lock_ = 0,
			.length = 0,
			.elements=NULL,
			.push = (void (*)(void *, void *)) Lista_push,
			.pop = (void *(*)(void *)) Lista_pop,
			.lock = (void (*)(void *)) Lista_lock,
			.unlock = (void (*)(void *)) Lista_unlock
	};
}