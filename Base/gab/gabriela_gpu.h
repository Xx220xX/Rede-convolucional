//
// Created by Xx220xX on 07/05/2020.
//
#define ALAN     0
#define TANH     1
#define RELU     2
#define SIGMOID  3
#define SOFTMAX  4
#define IDENTIFY 5
typedef struct {
    void *gab;
    unsigned int size;
} Gab;

void teste();

int create_DNN(Gab *p_gab, int *arq, int l_arq, int *funcs,char *norm,double hitLean);

void initGPU(const char *src);

void initWithFile(const char *filename);

void endGPU();



void call(Gab *p_gab, double *inp);

void learn(Gab *p_gab, double *trueOut);

void release(Gab *p_gab);

void getoutput(Gab *p_gab, double *out);

void setSeed(unsigned long int seed);

int randomize(Gab *p_gab);

int sethitlearn(Gab *p_gab, double hl);

int getA(Gab *p_gab, int l,double *det);

void checkLW();

void testXor(char *file);