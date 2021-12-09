//
// Created by hslhe on 22/09/2021.
//

#include "thread/Thread.h"

#include <windows.h>

void* default_newThread(void *func,void *arg){
	return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
}
int default_killThread(void * handle,int exit_code){
	return TerminateThread(handle,exit_code);
}
int default_releaseThread(void * handle){
	return CloseHandle(handle);
}
ManageThread Thread = {
		default_newThread,
		default_killThread,
		default_releaseThread

};

void setDefaultManageThread() {
	Thread = (ManageThread){
			default_newThread,
			default_killThread,
			default_releaseThread

	};
}

void setManageThread(vfvv newthread, ifvi killThread, ifv releaseThread) {
	Thread = (ManageThread){
		newthread,
		killThread,
		releaseThread
	};
}
