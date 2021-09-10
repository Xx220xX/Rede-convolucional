Gabriela IA
email: gab.cnn.ia@gmail.com
Versão 2.2.010

| Versão | Mudanças |
| ---- | :---- |
|2.2.014 | Adicionado camada PRelu  | 
|2.2.013 | Removido metódo corrige_peso das camadas, agora o metodo backpropagation faz a correção dos pesos caso learnable ==1   | 
|2.2.012 | O Tensor agora é possui a função randomize  | 
|2.2.011 | Adicionado função aleatoria com distribuição normal  | 
|2.2.010 | Adicionado função para treinar com Tensor  | 
|2.2.009 | Bug resolvido em kernel.h  | 
|2.2.008 | Tensores reformulados resolvendo problemas de compatibilidade funcionamento  | 
|2.2.007 | Mudançãs nos nomes das funções para adotar uma covenção | 
|2.2.006 | implementado thread de alta prioridade / tempo real | 
|2.2.005 | mudança no uso de thread | 
|2.2.004 | Retrocompatibilidade com arquivos de configuração da versão 2.1 | 
|2.2.003 | Nova configuração provida da vm Lua | 
|2.2.002 | Treino orientado a eventos | 
|2.2.001 | vm Lua inserida á rede, perda de compatibilidade com versões anteriores | 
|2.2.000 | Manage train criado | 
| #####  | Mudança na arquitetura do projeto |
|2.1.011 | Adicionado controle de memoria | 
|2.1.010 | Bugs corrigidos camada conv, pool e poolAv | 
|2.1.009 | Mudança no algoritmo de calculo de gradiente de entrada camada pooling |
|2.1.008 | Mudança no algoritmo de calculo de gradiente de entrada camada convolucional |
|2.1.007 | Otimização de calculo de gradiente dos pesos camada convolucional |
|2.1.006 | Agora os kernels podem ser compilados e executador somente pelo host |
|2.1.005 | O Tensor agora pode ser Alocado somente no HOST |
|2.1.004 | Suporte adicionado para SVM |
|2.1.003 | Removidos trabalhos sequencias das funções Kernel |
|2.1.002 | Bugs concertados em getValues |
|2.1.001 | Suporte a SVM removido | 
|2.1.000 | Revisado todas camadas, corrigido erros internos
| #####  | Mudança na arquitetura do projeto |
|2.0.017 | verificação de camadas |
|2.0.016 | verificação interna de erros adicionada |
|2.0.015 | Todas as camadas possui seus proprios parametros |
|2.0.014 | camada convNc adicionada |
|2.0.013 | camada polling av adicionada |
|2.0.012 | corrigido implementacao dropout |
|2.0.011 | camada padding corrigida |
|2.0.009 | camada padding adicionada |
|2.0.007 | camada conv corrigida |
| #####  | Mudança na arquitetura do projeto |
|1.0.010 | adicionado feed foward|
|1.0.009 | adicionado Cnn|
|1.0.008 | removido biblioteca sdl2|
|1.0.007 | removido vm python|
|1.0.006 | Adicionado biblioteca sdl2 |
|1.0.005 | Adicionado biblioteca conio2 |
|1.0.004 | Adicionado vm python|
|1.0.003 | Adicionado biblioteca OpenCL para programação paralela |
|1.0.002 | Adicionado biblioteca LCG RANDOM para pseudo-Aleatórios |
|1.0.001 | Tensor criado |
|1.0.000 | Rede convolucional criada |
