//
// Created by Henrique on 23-Jul-21.
//

#include "memory_utils.h"
#include "string.h"
#include "stdio.h"

struct pointer_size {
	void *p;
	size_t bytes;
};
typedef struct mem {
	struct pointer_size *memories;
	size_t len;
} MemStatus;

MemStatus memHost = {0}, memSVM = {0};

void putMem(MemStatus *mst, void *mem, size_t bytes) {
	mst->len += 1;
	mst->memories = realloc(mst->memories, sizeof(struct pointer_size) * mst->len);
	mst->memories[mst->len - 1] = (struct pointer_size) {mem, bytes};
}

int checkMem(MemStatus *mst, void *mem) {
	int i = mst->len - 1;
	for (; i >= 0; i--) {
		if (mst->memories[i].p == mem)
			break;
	}
	return i;
}

void removeMem(MemStatus *mst, void *mem) {
	int id = checkMem(mst, mem);
	if (id < 0)return;
	mst->len--;
	memmove(mst->memories + id, mst->memories + id + 1, (mst->len - id) * sizeof(struct pointer_size));
	mst->memories = realloc(mst->memories, mst->len * sizeof(struct pointer_size));
}

void *__alloc_mem_(size_t elements, size_t size_element) {
	void *p = calloc(elements, size_element);
	if (!p)return NULL;
	putMem(&memHost, p, elements * size_element);
	return p;
}

void __free_mem_(void *mem_pointer) {
	if (!mem_pointer)return;
	removeMem(&memHost, mem_pointer);
	free(mem_pointer);
}

void *__realloc_mem_(void *mem_pointer, size_t new_size) {
	if (mem_pointer) {
		removeMem(&memHost, mem_pointer);
	}
	void *p = realloc(mem_pointer, new_size);
	putMem(&memHost, p, new_size);
	return p;
}

void *__alloc_cl_svm__(cl_context context,
                       cl_svm_mem_flags flags,
                       size_t size,
                       cl_uint alignment) {
	void *p = clSVMAlloc(context, flags, size, alignment);
	if (!p)return NULL;
	putMem(&memSVM, p, size);
	return p;
}

void __free_cl_svm(cl_context context, void *svm_pointer) {
	if (!svm_pointer)return;
	removeMem(&memSVM, svm_pointer);
	clSVMFree(context, svm_pointer);
}

void printMemStatus() {
	size_t total = 0;
	char buff[250];
	printf("Host Memory\n   %zu memorias alocadas\n", memHost.len);
	for (int i = 0; i < memHost.len; i++) {
		struct pointer_size sp = memHost.memories[i];
		total += sp.bytes;
		printf("   %llX %s\n", (long long int) sp.p, printBytes(sp.bytes, buff));
	}
	printf("Shared Memory\n   %zu memorias alocadas\n", memSVM.len);
	for (int i = 0; i < memSVM.len; i++) {
		struct pointer_size sp = memSVM.memories[i];
		total += sp.bytes;
		printf("   %llX %s\n", (long long int) sp.p, printBytes(sp.bytes, buff));
	}

}

void releaseMemWatcher() {
	if (memSVM.memories)
		free(memSVM.memories);
	if (memHost.memories)
		free(memHost.memories);
}

char *printBytes(size_t bytes, char *buff) {
	unsigned int G, M, K, B;
	int b = 0;
	G = bytes / (1024 * 1024 * 1024);
	bytes %= (1024 * 1024 * 1024);
	M = bytes / (1024 * 1024);
	bytes %= (1024 * 1024);
	K = bytes / (1024);
	bytes %= (1024);
	B = bytes;
	if (G)
		b += sprintf(buff + b, "%uGB ", G);
	if (M)
		b += sprintf(buff + b, "%uMB ", M);
	if (K)
		b += sprintf(buff + b, "%uKB ", K);
	if (B || !b)
		b += sprintf(buff + b, "%uB ", B);

	buff[b - 1] = 0;
	return buff;
}
#ifdef MEMORY_WATCHER
#undef main
int main(int arg, char ** args) {
	printf("Running\n");
	int erro = __main__(arg,args);
	printMemStatus();
	releaseMemWatcher();
	return erro;
}
#define main __main__
#endif