#include<stdio.h>
#include<string.h>
#include<stdlib.h>
typedef struct {
	unsigned char * b;
	size_t len;
}Bytes;
void ppmp3(unsigned char *data, int x, int y, int z, char *fileName) {
	FILE *f = fopen(fileName, "w");
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", y, x);
	fprintf(f, "255\n");
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; ++j) {
			for (int k = 0; k < z; ++k) {
				fprintf(f, "%d", (int) (data[k*y*x+ i * y + j] ));
				if (k < z - 1)fprintf(f, " ");
			}
			if (j < y - 1)fprintf(f, " ");
		}
		if (i < x - 1)
			fprintf(f, "\n");
	}
	fclose(f);
}

long int findSize(char *file_name){
    // opening the file in read mode
	FILE* fp = fopen(file_name, "rb");  
    // checking if the file exist or not
	if (fp == NULL) {
		printf("File Not Found!\n");
		exit(-1);
	}
	fseek(fp, 0L, SEEK_END);

    // calculating the size of the file
	long int res = ftell(fp);
    // closing the file
	fclose(fp);

	return res;
}
Bytes loadbytes(char * fileName){
	Bytes bytes={0};
	bytes.len = findSize(fileName);
	FILE *f = fopen(fileName,"rb");
	bytes.b = (unsigned char *) calloc(sizeof(unsigned char),bytes.len);
	size_t load = fread(bytes.b,1,bytes.len,f);
	printf("%lld %lld \n", bytes.len,load);
	fclose(f);
	return bytes;
}
void saveImages(FILE *fimage,Bytes b,int n,int lenImg){
	unsigned char *data = b.b;
	for (int i = 0; i <n; ++i){
		data = data +1;
		fwrite(data,lenImg,1,fimage);
		data = data+lenImg;
	}

}
void saveLabel(FILE *flabel,Bytes b,int n,int lenImg){
	unsigned char *data = b.b;
	for (int i = 0; i <n; ++i){
		fwrite(data,1,1,flabel);
		data = data +1;
		data = data+lenImg;
	}
	
}

void extractData(char *filename,FILE *fimage,FILE *flabel){
	Bytes b = loadbytes(filename);
	saveImages(fimage,b,10000,32*32*3);
	saveLabel(flabel,b,10000,32*32*3);
	free(b.b);
}
int main(){
	FILE *fimage = fopen("imagesCifar10.ubyte","wb");
	FILE *flabel = fopen("labelsCifar10.ubyte","wb");
	extractData("cifar-10-batches-bin/data_batch_1.bin",fimage,flabel);
	extractData("cifar-10-batches-bin/data_batch_2.bin",fimage,flabel);
	extractData("cifar-10-batches-bin/data_batch_3.bin",fimage,flabel);
	extractData("cifar-10-batches-bin/data_batch_4.bin",fimage,flabel);
	extractData("cifar-10-batches-bin/data_batch_5.bin",fimage,flabel);
	extractData("cifar-10-batches-bin/test_batch.bin",fimage,flabel);
	fclose(fimage);
	fclose(flabel);
	return 0;
}