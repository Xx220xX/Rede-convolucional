//
// Created by Henrique on 10/12/2021.
//

#ifndef GAB_MATLAB_H
#define GAB_MATLAB_H

#include "string.h"

char _matlab_file_name_[292];

char *getName(const char *string, int withpath) {
	memset(_matlab_file_name_, 0, 292);
	strncpy(_matlab_file_name_, string, 250);
	int len = strlen(_matlab_file_name_);
	int p_ponto = len;
	int p_barra = 0;
	for (int i = len - 1; i >= 0; i--) {
		if (_matlab_file_name_[i] == '.') {
			p_ponto = i;
			_matlab_file_name_[p_ponto] = 0;
			break;
		}
	}
	snprintf(_matlab_file_name_ + p_ponto, 292 - p_ponto, ".m");
	for (int i = p_ponto - 1; i >= 0; i--) {
		if (_matlab_file_name_[i] == '\\' || _matlab_file_name_[i] == '/') {
			p_barra = i + 1;
			break;
		}
	}

	if (withpath) {
		char *path = "../matlab/";
		memmove(_matlab_file_name_ + 10, _matlab_file_name_ + p_barra, p_ponto - p_barra + 2);
		memset(_matlab_file_name_ + 10 + p_ponto - p_barra + 2, 0, 1);
		memcpy(_matlab_file_name_, path, 10);

	} else {
		memmove(_matlab_file_name_, _matlab_file_name_ + p_barra, p_ponto - p_barra + 2);
		memset(_matlab_file_name_ + p_ponto - p_barra + 2, 0, 1);
	}
//	printf("%s\n",_matlab_file_name_);
	return _matlab_file_name_;
}

#define matlabInit()FILE *_matlab_file_ = fopen(getName(__FILE__,1),"w");matlab("clc;clear all;close all;")

#define matlabf(format, ...) fprintf(_matlab_file_,format,##__VA_ARGS__);
#define matlab(x) fprintf(_matlab_file_,"%s\n",x)
#define Tmatlab(tensor, name) if(tensor)_tomatlab(tensor,_matlab_file_, name, NULL)
#define Trmatlab(tensor, name)if(tensor)_tomatlab(tensor,_matlab_file_, name, "mshape")
#define Tsmatlab(tensor, name, shape, ...)if(tensor)_tomatlab(tensor,_matlab_file_, name,NULL);matlabf(name" = reshape("name","shape");",##__VA_ARGS__)


#define matlabCmp(x, y)matlab("figure"); \
                                                  \
matlab("for i = 1:size("x",3)");                  \
matlab("	subplot(size("x",3),1,i);hold on;");     \
matlab(" tmp = [];\n"\
"  for j=1:size("x",1)\n"\
"    tmp =[tmp "x"(j,:,i)];\n"\
"  end\n");                                         \
matlab("	plot(tmp)");\
matlab(" tmp = [];\n"\
"  for j=1:size("y",1)\n"\
"    tmp =[tmp "y"(j,:,i)];\n"\
"  end\n");                                         \
matlab("	plot(tmp)");\
matlab("end");\
matlab("suptitle(sprintf('"x" vs "y" erro = %f',var(("x"(:) - "y"(:)))));");\
matlab("legend('"x"','"y"');");\
matlab("if gcf() == 1");\
matlabf("print('%s.pdf')",getName(__FILE__,1));                     \
matlabf("else print('%s.pdf', '-append'); end;",getName(__FILE__,1))

#define  matlabAtivation(y, x, fid) \
    switch (fid) {\
        case 0:\
            matlab(y" = 1.0 ./ (1.0 + exp(-"x"));");\
            break;\
        case 1:\
            matlab(y" = 1.0 ./ (1.0 + exp(-"x"));");\
            matlab(y" = "y" .* (1.0  -" y");");\
            break;\
        case 2:                                             \
             matlab(y" = tanh("x");");                      \
            break;                                          \
        case 3:                                             \
            matlab(y" = 1.0 -  tanh("x") .^2;");            \
            break;                                          \
        case 4:                                             \
            matlabf(y " = ("x"> 0).* "x" *%.12f +("x"< 0).* "x"*%.12f ;\n",cf->fa.greater,cf->fa.less);                 \
            break;                                          \
        case 5:                                             \
        matlabf(y " = ("x"> 0).*%.12f +("x"< 0).*%.12f ;\n",cf->fa.greater,cf->fa.less);            \
            break;                                          \
        case FLIN:                                             \
            matlab(y " = "x";");                            \
            break;                                          \
        case 7:                                             \
            matlab(y " = 1;");                              \
            break;                                          \
        case 8:                                             \
            matlab(y " = alan("x");");                      \
            break;                                          \
        case 9:                                             \
            matlab(y " = dfalan("x");");                    \
            break;                  \
        case FSOFTMAX:                    \
            matlab("maximom = max("x");");\
            matlab("epm = exp("x"-maximom);");\
            matlab("somam = sum(epm);");\
            matlab(y" = epm/somam;");\
            matlabf(y"("y">%.12f) = %.12f;\n",1- cf->fa.epsilon,1- cf->fa.epsilon);\
            matlabf(y"("y"<%.12f) = %.12f;\n",cf->fa.epsilon,cf->fa.epsilon);\
            break;                     \
        case 13:                    \
            matlab(y" = 1;");\
            break;                            \
        default:                                            \
        fprintf(stderr,"Função de ativação desconhecida\n"); \
    }
#define matlabEnd()fclose(_matlab_file_)


#include <stdio.h>
#include <windows.h>
#include <tlhelp32.h>

BOOL CALLBACK EnumWindowsProcMy(HWND hwnd, LPARAM lParam) {
	DWORD lpdwProcessId;
	GetWindowThreadProcessId(hwnd, &lpdwProcessId);
	if (lpdwProcessId == lParam) {
//g_HWND = hwnd;
		SwitchToThisWindow(hwnd, TRUE);
		return FALSE;
	}
	return TRUE;
}

int IsProcessRunning(const char *processName) {
	int exists = 0;
	PROCESSENTRY32 entry;
	entry.dwSize = sizeof(PROCESSENTRY32);

	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	int len = strlen(processName);
	if (Process32First(snapshot, &entry)) {
		while (Process32Next(snapshot, &entry)) {
			if (!strncmp(processName, entry.szExeFile, len)) {
				exists = 1;
				printf("%s\n", entry.szExeFile);
//				SwitchToThisWindow(GetWindowThreadProcessId(), bool fAltTab)
				EnumWindows(EnumWindowsProcMy, entry.th32ProcessID);
				Sleep(100);
				break;
			}
		}
	}

	CloseHandle(snapshot);
	return exists;
}


#endif //GAB_MATLAB_H
