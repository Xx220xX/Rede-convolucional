//
// Created by hslhe on 11/08/2021.
//

#ifndef CNN_GPU_STRING_H
#define CNN_GPU_STRING_H

#include <string.h>

typedef struct {
	char *d;
	size_t size;
	char release;
} String;

String Strf(char *format, ...);

String vStrf(char *format, va_list args);

String StrS(String sformat, ...);

String vStrS(String sformat, va_list args);

void releaseStr(String *s);
#define StrClearAndCopy(dst,char_src)releaseStr(&(dst));(dst) = Strf("%s",char_src)
#endif //CNN_GPU_STRING_H
