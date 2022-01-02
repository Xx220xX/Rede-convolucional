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
	return _matlab_file_name_;
}

#define matlabInit()FILE *_matlab_file_ = fopen(getName(__FILE__,1),"w");matlab("clc;clear all;close all;")

#define matlabf(format, ...) fprintf(_matlab_file_,format,##__VA_ARGS__);fprintf(_matlab_file_,"\n")
#define matlab(x) fprintf(_matlab_file_,"%s\n",x)
#define Tmatlab(tensor, name) _tomatlab(tensor,_matlab_file_, name, NULL)
#define Trmatlab(tensor, name)_tomatlab(tensor,_matlab_file_, name, "mshape")
#define Tsmatlab(tensor, name, shape, ...)_tomatlab(tensor,_matlab_file_, name,NULL);matlabf(name" = reshape("name","shape");",##__VA_ARGS__)


#define matlabCmp(x, y)matlab("figure"); \
                                                  \
matlab("for i = 1:size("x",3)");                  \
matlab("	subplot(size("x",3),1,i);hold on;");\
matlab("	plot("x"(:,:,i)(:))");\
matlab("	plot("y"(:,:,i)(:))");      \
matlab("end");\
matlab("suptitle(sprintf('"x" vs "y" erro = %f',var(("x" - "y")(:))));");\
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
            matlab(y " = ("x"> 0).* "x";");                 \
            break;                                          \
        case 5:                                             \
            matlab(y " = ("x"> 0).* 1;");                   \
            break;                                          \
        case 6:                                             \
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
            break;                                          \
        default:                                            \
        fprintf(stderr,"Função de ativação desconhecida\n"); \
    }
#define matlabEnd()fclose(_matlab_file_)
#endif //GAB_MATLAB_H
