//
// Created by Henrique on 4/2/2021.
//

#ifndef CNN_GPU_UTEISTREINO_H
#define CNN_GPU_UTEISTREINO_H

#include <pthread.h>
#include"vectorUtils.h"

typedef struct {
	FILE *jsEpoca;
	FILE *jsErro;
	FILE *jsAcerto;
	double *epoca;
	double *erro;
	double *acerto;
	int len;
	pthread_t tid;
} SalveJsArgs;

void *salveJS(SalveJsArgs *args) {
	for (int i = 0; i < args->len; i++) {
		fprintf(args->jsEpoca, "%lf,", args->epoca[i]);
		fprintf(args->jsErro, "%lf,", args->erro[i]);
		fprintf(args->jsAcerto, "%lf,", args->acerto[i]);
	}
	free_mem(args->epoca);
	free_mem(args->erro);
	free_mem(args->acerto);
	return NULL;
}

/*
void salveCnnOutAsPPM(const char *name, Cnn c) {
	FILE *f = fopen(name, "wb");
	int maxH = 0;
	int maxW = 0;
	int w, h;
	double *dt;
	Tensor t;
	unsigned char px;
	for (int cm = 0; cm < c->size; cm++) {
		t = c->camadas[cm]->saida;
		w = t->y;
		h = t->x;
		if (t->x > t->y) {
			w = t->x;
			h = t->y;
		}
//		w = w * t->z;
		w = w * t->z + t->z - 1;
		maxH += h;
		if (maxW < w)maxW = w;
	}
	maxW += 2;
	maxH = maxH + c->size;
	fprintf(f, "P5 ");
	fprintf(f, "%d %d ", maxW, maxH);
	fprintf(f, "255 ");

	char *image = calloc(maxW, maxH);
	int imi = 0, imj = 0;
	for (int cm = 0; cm < c->size; cm++) {
		t = c->camadas[cm]->saida;
		w = t->y;
		h = t->x;
		if (t->x > t->y) {
			w = t->x;
			h = t->y;
		}
		dt = calloc(t->bytes, 1);
		TensorGetValues(c->queue, t, dt);
		normalizeGPU(c, dt, dt, t->bytes / sizeof(double), 255, 0);
		for (int i = 0; i < h; i++) {
			imj = 1;
			//image[imi*maxW+imj++] = 0xff;
			for (int z = 0; z < t->z; ++z) {
				for (int j = 0; j < w; ++j) {
					px = ((unsigned int) (dt[z * t->x * t->y + i * t->y + j])) & 0xff;
					image[imi * maxW + imj++] = px;

				}
				//image[imi*maxW+imj++]=0xff;
				imj++;
			}
			imi++;

		}

		//memset(image+imi*maxW,255,maxW);
		imi++;
		free_mem(dt);
	}
	fwrite(image, maxH, maxW, f);
	free_mem(image);
	fclose(f);
}
*/
const char INDEX_HTML[] =
		"<!doctype html>\n"
		"<html lang=\"pt-br\">\n"
		"<head>\n"
		"\t<meta charset=\"utf-8\">\n"
		"\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
		"\t<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\"\n"
		"\tintegrity=\"sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC\" crossorigin=\"anonymous\">\n"
		"\t<script src=\"https://polyfill.io/v3/polyfill.min.js?features=es6\"></script>\n"
		"\t<script type=\"text/javascript\" id=\"MathJax-script\" async src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js\"/>\n"
		"\t<script src=\"https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js\"\n"
		"\tintegrity=\"sha384-IQsoLXl5PILFhosVNubq5LC7Qb9DXgDA9i+tQ8Zj3iwWAwPtgFTxbJ8NT4GN1R8p\"\n"
		"\tcrossorigin=\"anonymous\"></script>\n"
		"\t<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.min.js\"\n"
		"\tintegrity=\"sha384-cVKIPhGWiC2Al4u+LWgxfKTRIcfu0JTxR+EQDz/bgldoEyl4H0zUF0QKbrJ0EcQF\"\n"
		"\tcrossorigin=\"anonymous\"></script>\n"
		"\t<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>\n"
		"\t\n"
		"\t<title>Treino</title>\n"
		"</head>\n"
		"<body class=\"d-flex justify-content-md-center\">\n"
		"\t<div class=\"container\">\n"
		"\t\t<div >\n"
		"\t\t\t<h5 class=\"alert alert-primary\" role=\"alert\">Arquitetura</h5>\n"
		"\t\t\t<p></p>\n"
		"\t\t\t<table class = \"table table-sm \" id=\"tabela_arquitetura\"></table>\n"
		"\t\t</div>\n"
		"\n"
		"\t\t<h5 class=\"alert alert-primary\" role=\"alert\">Treinamento</h5>\n"
		"\t\t<div class=\"container\">\n"
		"\t\t\t<div class=\"row\" style=\"height:500px\">\n"
		"\t\t\t\t<div class=\"col\" >\n"
		"\t\t\t\t\t<div id=\"graficoErro\" style=\"height:500px\" ></div>\n"
		"\t\t\t\t</div>\n"
		"\t\t\t\t<div class=\"col\">\n"
		"\t\t\t\t\t<div id=\"graficoAcerto\" style=\"height:500px\" ></div>\n"
		"\t\t\t\t</div>\n"
		"\n"
		"\t\t\t</div>\n"
		"\t\t</div>\n"
		"\t\t\n"
		"\t\t<div >\n"
		"\t\t\t<h5 class=\"alert alert-primary\" role=\"alert\">Avaliação</h5>\n"
		"\t\t\t<p></p>\n"
		"\t\t\t<table class = \"table table-sm\" id=\"tabela_fitnes\">\n"
		"\t\t\t</table>\n"
		"\t\t\t\n"
		"\t\t</div>\n"
		"\t</div>\n"
		"</body>\n"
		"<script type=\"text/javascript\" >\n"
		"\tfunction plot(elementID,x,y,xlabel,ylabel,title='',curve_name='',xlim=[null,null],ylim=[null,null]) {\n"
		"\t\tgoogle.charts.load('current', {'packages':['corechart']});\n"
		"\t\tgoogle.charts.setOnLoadCallback(drawChart);\n"
		"\t\tlet element = document.getElementById(elementID);\n"
		"\n"
		"\t\tfunction drawChart(){\n"
		"\t\t\tconsole.log(\"here\\n\");\n"
		"\t\t\tlet dt = [['',curve_name]];\n"
		"\t\t\tfor (let i in x){\n"
		"\t\t\t\tdt.push([x[i],y[i]]);\n"
		"\t\t\t}\n"
		"\n"
		"\t\t\tlet data = google.visualization.arrayToDataTable(dt);\n"
		"\n"
		"\t\t\tlet options = {\n"
		"\t\t\t\ttitle: title,\n"
		"\t\t\t//curveType: 'function',\n"
		"\t\t\tlegend: { position: 'top' },\n"
		"\t\t\thAxis:{title:xlabel,viewWindow:{min:xlim[0],max:xlim[1]}},\n"
		"\t\t\tvAxis:{title:ylabel,viewWindow:{min:ylim[0],max:ylim[1]}},\n"
		"\t\t\t\n"
		"\t\t};\n"
		"\t\tlet chart = new google.visualization.LineChart(element);\n"
		"\t\tchart.draw(data, options);\n"
		"\t\t\n"
		"\t}\n"
		"}\n"
		"//document.getElementById(\"numeroCamadas\").innerHTML = 5;\n"
		"var tabela_arquitetura = document.getElementById(\"tabela_arquitetura\");\n"
		"var tabela_fitnes = document.getElementById(\"tabela_fitnes\");\n"
		"function tablePutColum(table,colum){\n"
		"\tlet st = \t'<tr>';\n"
		"\tlet c;\n"
		"\tfor(let cl in colum){\n"
		"\t\tc = '<th scope=\"col\" aligned=\"center\">'+ colum[cl] +'</th>';\n"
		"\t\tst +=\tc;\n"
		"\t}\n"
		"\tst +=\t'<tr>\\n';\n"
		"\ttable.innerHTML +=st;\n"
		"}\n"
		"</script>\n"
		"<script type=\"text/javascript\" src=\"js/camada.js\"></script>\n"
		"<script type=\"text/javascript\" src=\"js/fitnes.js\"></script>\n"
		"<script type=\"text/javascript\" src=\"js/dataEpoca.js\"></script>\n"
		"<script type=\"text/javascript\" src=\"js/dataErro.js\"></script>\n"
		"<script type=\"text/javascript\" src=\"js/dataAcerto.js\"></script>\n"
		"</html>";
#endif //CNN_GPU_UTEISTREINO_H
