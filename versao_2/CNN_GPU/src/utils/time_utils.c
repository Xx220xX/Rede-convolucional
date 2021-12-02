//
// Created by Henrique on 28-Jul-21.
//

#include "utils/time_utils.h"
#include <windows.h>

double getns() {
	return getus() * 1e3;
}

double getus() {
	FILETIME ft;
	LARGE_INTEGER li;
	GetSystemTimePreciseAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	u_int64 ret = li.QuadPart;
//	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	return (double) ret / 10.0;
}

double getms() {
	return getus() * 1.0e-3;
}

double getsec() {
	return getus() * 1e-6;
}