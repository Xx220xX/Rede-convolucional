#include <conio.h>
#include <windows.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include "conio2/conio2.h"
#include "mainui.h"
#include "utils/time_utils.h"

int main(int arg, char **args) {
	system("chcp 65001");
	showVersion();
	if (arg != 2) {
		fprintf(stderr, "É esperado um arquivo lua para executar o programa\n");
		return -1;
	}
	unsigned int escolha;
	char *file = args[1];
	ManageTrain manageTrain = createManageTrain(file, 0.1, 0.0, 0.0);
	if (manageTrain.cnn->error.error)goto end;
	manage2WorkDir(&manageTrain);

	ManageTrainSetEvent(manageTrain.OnfinishEpic, OnfinishEpic);
	ManageTrainSetEvent(manageTrain.OnfinishFitnes, OnfinishFitnes);
	ManageTrainSetEvent(manageTrain.UpdateTrain, UpdateTrain);
	ManageTrainSetEvent(manageTrain.UpdateFitnes, UpdateFitnes);
	ManageTrainSetEvent(manageTrain.UpdateLoad, UpdateLoad);
	resetDir("statistic");
	createDir("redes");
	createDir("tabela");

	// ler imagens
	ManageTrainloadImages(&manageTrain, 1);
	manageTrainLoop(&manageTrain, 0);

	treinar:
	manageTrainSetRun(&manageTrain, 1);
	ManageTraintrain(&manageTrain, 1);
	manageTrainLoop(&manageTrain, 0);

	fitnes:
	manageTrainSetRun(&manageTrain, 1);
	// Fitness
	ManageTrainfitnes(&manageTrain, 1);
	manageTrainLoop(&manageTrain, 0);
	printf("\nO treinamento terminou:\n");
	const char *tmpfile = "tmp.lua";
	char buf[250] = "";
	escolha = 8;
	double t0 = getms();
	printf("pressione qualquer tecla para abrir o menu:...\n");
	printf("Encerra em  s");
	int y = wherey();

	while (getms() - t0 < 5000) {
		if (kbhit()) {
			escolha = 0;
			break;
		}
		gotoxy(12, y);
		printf("%d", (int)( 5-(getms() - t0) / 1000));
	}
	printf("\n");
	if (escolha != 8)
		system("pause");
	else{
			FILE *f = fopen("redes/t1.cnn", "wb");
			cnnSave(manageTrain.cnn, f);
			fclose(f);
	}
	fflush(stdin);
	while (escolha != 8) {

		system("cls");
		gotoxy(1, 1);
		printf("[1] continuar treinamento\n");
		printf("[2] continuar avaliação \n");
		printf("[3] salvar rede\n");
		printf("[4] mostrar rede:\n");
		printf("[5] lua console:\n");
		printf("[6] resetar imagens\n");
		printf("[7] show graphic\n");
		printf("[8] encerrar\nopcao:");
		scanf("%u", &escolha);
		fflush(stdin);

		switch (escolha) {
			case 1:
				goto treinar;
			case 2:
				goto fitnes;
			case 3:
				printf("nome :");
				int b = snprintf(buf,250,"redes/");
				fgets(buf+b, 250-b, stdin);
				FILE *f = fopen(buf, "w");
				cnnSave(manageTrain.cnn, f);
				fclose(f);
				printf("salvado\n");
				break;
			case 4:
				printCnn(manageTrain.cnn);
				system("pause");
				break;
			case 5:
				CnnLuaConsole(manageTrain.cnn);
				break;
			case 6:
				manageTrain.image = 0;
				break;
			case 7:
				system("python showGraphic.py");
						break;
			case 8:
				break;
			default:
				printf("invalid option");

		}

		Sleep(300);
	}

	end:
	getClError(manageTrain.cnn->error.error, manageTrain.cnn->error.msg, EXCEPTION_MAX_MSG_SIZE);
	printf("%d %s\n", manageTrain.cnn->error.error, manageTrain.cnn->error.msg);
	releaseManageTrain(&manageTrain);
	return 0;
}



