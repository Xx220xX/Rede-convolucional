//
// Created by hslhe on 15/08/2021.
//

#ifndef CNN_GPU_THREAD_H
#define CNN_GPU_THREAD_H

#include <windows.h>

typedef HANDLE Handle;

typedef void *(*vfvv)(void *, void *);

typedef int (*ifvi)(void *, int);

typedef int (*ifv)(void *);

typedef struct {
	vfvv newThread;
	ifvi killThread;
	ifv releaseThread;
} ManageThread;

extern ManageThread Thread;

void setDefaultManageThread();

void setManageThread(vfvv newthread, ifvi killThread, ifv releaseThread);


#define Thread_new(func, arg) Thread.newThread(func,arg)
#define ThreadKill(handle, exit_code) Thread.killThread(handle,exit_code)
#define Thread_Release(handle) Thread.releaseThread(handle)

//#define newThreadSuspend(func, arg, id) CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, CREATE_SUSPENDED, &(id))
//#define ThreadResume(handle)ResumeThread(handle)
//#define ThreadSuspend(handle)SuspendThread(handle)


#endif //CNN_GPU_THREAD_H
