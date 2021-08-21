//
// Created by hslhe on 15/08/2021.
//

#ifndef CNN_GPU_THREAD_H
#define CNN_GPU_THREAD_H

#include <windows.h>
typedef HANDLE Thread;
#define newThread(func, arg, id) CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, id)
#define newThreadSuspend(func, arg, id) CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, CREATE_SUSPENDED, &(id))
#define ThreadResume(handle)ResumeThread(handle)
#define ThreadSuspend(handle)SuspendThread(handle)
#define ThreadKill(handle, exit_code)TerminateThread(handle,exit_code)
#define ThreadClose(handle) CloseHandle(handle)

#endif //CNN_GPU_THREAD_H
