#include <conio.h>
#include <windows.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include "conio2/conio2.h"
#include "mainui.h"

int main(int arg, char **args) {
	system("chcp 65001");
	showVersion();
	if (arg != 2) {
		fprintf(stderr, "É esperado um arquivo lua para executar o programa\n");
		return -1;
	}
	char *file = args[1];
	ManageTrain manageTrain = createManageTrain(file, 0.1, 0.0, 0.0);
	if (manageTrain.cnn->error.error)goto end;
	manage2WorkDir(&manageTrain);

	ManageTrainSetEvent(manageTrain.OnfinishEpic, OnfinishEpic);
	ManageTrainSetEvent(manageTrain.OnfinishFitnes, OnfinishFitnes);
	ManageTrainSetEvent(manageTrain.UpdateTrain, UpdateTrain);
	ManageTrainSetEvent(manageTrain.UpdateFitnes, UpdateFitnes);
	ManageTrainSetEvent(manageTrain.UpdateLoad, UpdateLoad);

	// ler imagens
	ManageTrainloadImages(&manageTrain, 1);
	manageTrainLoop(&manageTrain, 0);
	// Treinar
//	ManageTraintrain(&manageTrain, 1);
//	manageTrainLoop(&manageTrain, 0);
	printf("treinou \n");
	manageTrainSetRun(&manageTrain, 1);
	// Fitness
	ManageTrainfitnes(&manageTrain, 1);
	manageTrainLoop(&manageTrain, 0);


//	FILE *f = fopen("t1.cnn", "wb");
//	cnnSave(manageTrain.cnn, f);
//	fclose(f);
	end:
	getClError(manageTrain.cnn->error.error, manageTrain.cnn->error.msg, EXCEPTION_MAX_MSG_SIZE);
	printf("%d %s\n", manageTrain.cnn->error.error, manageTrain.cnn->error.msg);
	releaseManageTrain(&manageTrain);
	system("pause");
	return 0;
}



