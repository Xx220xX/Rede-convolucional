//
// Created by hslhe on 01/09/2021.
//

#ifndef CNN_GPU_COMMONSTEST_H
#define CNN_GPU_COMMONSTEST_H

#define SEED 99

#define GetTIME() getms()
#define TIME_UNIT "s"
typedef struct {
	double aux;
	double init;
	double create_vars;
	double putvalues;
	double mult;
	double all;
} Time_proces;

void printTime(Time_proces t, const char *m) {

	printf("| "

	);
	printf("%sCriar variveis %lf "TIME_UNIT
	"\nColocar valores %lf "TIME_UNIT
	"\nMultiplicacao %lf "TIME_UNIT
	"\nTotal %lf "TIME_UNIT
	"\n\n",
	m, t.create_vars, t.putvalues, t.mult, t.all);
}

#endif //CNN_GPU_COMMONSTEST_H
