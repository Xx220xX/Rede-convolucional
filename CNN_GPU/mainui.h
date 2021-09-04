#include <conio.h>
#include <windows.h>
#include "utils/manageTrain.h"
#include "utils/vectorUtils.h"
#include "conio2/conio2.h"

// call backs
void onLoad(ManageTrain *t);

void OnfinishEpic(ManageTrain *t);

void OnInitTrain(ManageTrain *t);

void OnfinishTrain(ManageTrain *t);

void OnInitFitnes(ManageTrain *t);

void OnfinishFitnes(ManageTrain *t);

void UpdateTrain(ManageTrain *t);

void UpdateFitnes(ManageTrain *t);

void onLoad(ManageTrain *t) {
    printf("Imagens carregadas com sucesso\n");
    Sleep(400);
}

void OnfinishEpic(ManageTrain *t) {
//	printf("Epoca: \n");
}

void OnInitTrain(ManageTrain *t) {
    printf("treino iniciado\n");

}

void OnfinishTrain(ManageTrain *t) {
    printf("Treino finalizado\n");
}

void UpdateTrain(ManageTrain *mt) {
    double imps = 0;
    Estatistica *t =(Estatistica *) mt;
    if (t->tr_time)
        imps = (t->tr_epoca_atual * t->tr_numero_imagens + t->tr_imagem_atual) / (double) t->tr_time * 1000.0;
    size_t tmp_restante_epoca = round((t->tr_numero_imagens - t->tr_imagem_atual - 1) / imps);
    size_t tmp_restante_treino =
            round(((t->tr_numero_epocas - t->tr_epoca_atual - 1) * t->tr_numero_imagens + t->tr_numero_imagens -
                   t->tr_imagem_atual - 1) / imps);

    char *format =
            "Epoca %d of %d     imagem %d of %d \n"
            "Tempo estimado final do treino  %lld:%02lld:%02lld\n"
            "Tempo estimado final da epoca %lld:%02lld:%02lld\n"
            "Imagens por segundo %.2lf\n"
            "Mse %lf\n"
            "Acerto medio %lf\n";

    gotoxy(1,1);
    int bytes = printf("Epoca %d of %d     imagem %d of %d \n",t->tr_epoca_atual,t->tr_numero_epocas,t->tr_imagem_atual,t->tr_numero_imagens);
    bytes += printf("Tempo estimado final do treino  %lld:%02lld:%02lld\n",
                    tmp_restante_treino / 3600,
                    (tmp_restante_treino % 3600) / 60,
                    (tmp_restante_treino % 3600) % 60);
    bytes += printf("Tempo estimado final da epoca %lld:%02lld:%02lld\n",
                    tmp_restante_epoca / 3600,
                    (tmp_restante_epoca % 3600) / 60,
                    (tmp_restante_epoca % 3600) % 60);
    bytes += printf("Imagens por segundo %.2lf\n",
                    imps);
    bytes += printf("Mse %lf\n"
                    "Acerto medio %lf\n",
                    t->tr_erro_medio,
                    t->tr_acerto_medio*100);

    static int delete = 0;
    delete = bytes - delete;
    for(int i=0;i<delete;i++)
        printf(" ");
    delete = bytes;
    char c = 0;
    if (kbhit()) {
        c = getche();
        if (c == 'q') {
            manageTrainSetRun((ManageTrain *) t, 0);
        }
    }
    Sleep(100);
//	system("cls");
}

void UpdateLoad(ManageTrain *t){
    gotoxy(1,1);
    printf("Imagem: %lld\n", t->et.ld_imagem_atual);
    printf("Label: %lld", t->et.ll_imagem_atual);

}
