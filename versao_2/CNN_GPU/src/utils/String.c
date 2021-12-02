//
// Created by hslhe on 11/08/2021.
//

#include "cnn/utils/String.h"
#include <stdarg.h>
#include <stdio.h>
#include "utils/memory_utils.h"

String Strf(char *format, ...) {
	va_list arg;
	va_start(arg, format);
	String result = vStrf(format, arg);
	va_end(arg);
	return result;
}

String vStrf(char *format, va_list args) {
	String result;
	result.size = vsnprintf(NULL, 0, format, args) + 1;
	result.d = alloc_mem(result.size, 1);
	vsnprintf(result.d, result.size, format, args);
	result.release = 1;
	return result;
}

void releaseStr(String *s) {
	if (!s)return;
	if (s->release && s->d) {
		free_mem(s->d);
		s->d = NULL;
		s->size = 0;
	}
}

String StrS(String sformat, ...) {
	va_list arg;
	va_start(arg, sformat);
	String result = vStrf(sformat.d, arg);
	va_end(arg);
	return result;
}

String vStrS(String sformat, va_list args) {
	String result;
	result.size = vsnprintf(NULL, 0, sformat.d, args) + 1;
	result.d = alloc_mem(result.size, 1);
	vsnprintf(result.d, result.size, sformat.d, args);
	result.release = 1;
	return result;
}



char *vmprintf(char *format, va_list list) {
	char *result = NULL;
	int len = vsprintf(result, format, list);
	result = alloc_mem(len + 1, sizeof(char *));
	vsprintf(result, format, list);
	return result;

}

char *mprintf(char *format, ...) {
	va_list list;
	va_start(list, format);
	char *result = vmprintf(format, list);
	va_end(list);
	return result;

}

