//
// Created by hslhe on 09/11/2021.
//

#ifndef CNN_GPU_LOG_H
#define CNN_GPU_LOG_H



#define FLOGF(file, format, ...){FILE *f = fopen(file,"a");fprintf(f,format,## __VA_ARGS__);fprintf(f,"\n");fclose(f);}
#define LOGF(...) FLOGF("logs/logcnn.txt",##__VA_ARGS__)



#endif //CNN_GPU_LOG_H
