#include <conio.h>
#include <windows.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include "conio2/conio2.h"
#include "mainui.h"
#include "utils/time_utils.h"
#include "utils/log.h"

atomic_int *run;
int executing = 1;

static BOOL WINAPI console_ctrl_handler(DWORD dwCtrlType);


int main(int arg, char **args) {
	system("chcp 65001");
	LOGF("main:init");
	showVersion();
	if (arg != 2) {
		fprintf(stderr, "É esperado um arquivo lua para executar o programa\n");
		return -1;
	}
	unsigned int escolha;
	char *file = args[1];
	ManageTrain manageTrain = createManageTrain(file, 0.1, 0.0, 0.0, 0);
	SetConsoleCtrlHandler((PHANDLER_ROUTINE) (console_ctrl_handler), TRUE);
	run = &manageTrain.can_run;
	if (manageTrain.cnn->error.error)goto end;
	manage2WorkDir(&manageTrain);

	ManageTrainSetEvent(manageTrain.OnfinishEpic, OnfinishEpic);
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

	ManageTraintrain(&manageTrain, 0);
	manageTrainLoop(&manageTrain, 0);

	fitnes:

	// Fitness
	ManageTrainfitnes(&manageTrain, 0);
	manageTrainLoop(&manageTrain, 0);

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
			printf("%d", (int) (5 - (getms() - t0) / 1000));
		}
	printf("\n");
	if (escolha != 8)
		system("pause");
	else {
		FILE *f = fopen("redes/t1.cnn", "wb");
		cnnSave(manageTrain.cnn, f);
		fclose(f);
	}
	fflush(stdin);
	while (escolha != 8) {
		manageTrain.cnn->error.error = 0;
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
				manageTrainSetRun(&manageTrain, 1);
				goto treinar;
			case 2:
				manageTrainSetRun(&manageTrain, 1);
				goto fitnes;
			case 3:
				printf("nome :");
				int b = snprintf(buf, 250, "redes/");
				fgets(buf + b, 250 - b, stdin);
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
	executing = 0;
	LOGF("main:end");
	return 0;
}


void onForceEnd() {
	LOGF("onForceEnd:init");

	if (!run) {
		printf("Finalizando\n");
		return;
	}
	*run = 0;
	Sleep(100);
	printf("Finalizando\n");
	while (executing) {
		Sleep(10);
	}
	printf("Finalizado!!!\n");
	LOGF("onForceEnd:end");
	return;
}

static BOOL WINAPI console_ctrl_handler(DWORD dwCtrlType) {
	LOGF("console_ctrl_handler:init");
	switch (dwCtrlType) {
		case CTRL_C_EVENT: // Ctrl+C
			onForceEnd();
			break;
		case CTRL_BREAK_EVENT: // Ctrl+Break
			break;
		case CTRL_CLOSE_EVENT: // Closing the console window
			onForceEnd();
			break;
		case CTRL_LOGOFF_EVENT: // User logs off. Passed only to services!
		LOGF("console_ctrl_handler:end");
			return TRUE;
			break;
		case CTRL_SHUTDOWN_EVENT: // System is shutting down. Passed only to services!
			onForceEnd();
			break;
	}
	// Return TRUE if handled this message, further handler functions won't be called.
	// Return FALSE to pass this message to further handlers until default handler calls ExitProcess().
	LOGF("console_ctrl_handler:end");
	return FALSE;
}
