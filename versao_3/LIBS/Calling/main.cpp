#include <iostream>
#include <cstdarg>
#include <windows.h>
#include <tuple>
#include <utility>
#include <stdio.h>

void t(int a, int b) {
	std::cout << a << " " << b << std::endl;
}


int first() {
	__builtin_return(
			__builtin_apply(
					(void (*)()) second, __builtin_apply_args(), 512));
}
int main() {
//	size_t bytes = sizeof (int)*4;
//	void *args = calloc(bytes,1);
//	int a = 55,b=10;
//	memcpy(args,&a,sizeof(int));
//	memcpy(args,&b,sizeof(int));
//	void (*f)(...);
//	f =(void (*)(...))t;
//	__builtin_apply(f,args,16);
//

}


