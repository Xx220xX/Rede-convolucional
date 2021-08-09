
typedef struct {
	double *tr_mse_vector;
	double *tr_acertos_vector;
	UINT tr_imagem_atual;
	UINT tr_numero_imagens;
	UINT tr_epoca_atual;
	UINT tr_numero_epocas;
	double tr_erro_medio;
	double tr_acerto_medio;
		
	UINT ft_imagem_atual;
	UINT ft_numero_imagens;
	double * ft_erro_medio;
	double * ft_acerto_medio;
	UINT ft_numero_classes;

} Estatistica;
