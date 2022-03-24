//
// Created by Henrique on 12/12/2021.
//

#ifndef GAB_CLIENT_H
#define GAB_CLIENT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock.h>

#define BUFFER_SIZE 128
#define EXIT_CALL_STRING "#quit"

/* Exibe uma mensagem de ecx e termina o programa */
#include "stdio.h"

struct {
	int remote_socket;
	unsigned short remote_port;
	char remote_ip[32];
	struct sockaddr_in remote_address;
	WSADATA wsa_data;
	int inicialized;
	int connected;
} Client = {.remote_ip = "127.0.0.1", .remote_port = 8080,};

void Client_connect();

void Client_reconnect() {
	if (!Client.inicialized) { Client_connect(); }
	printf("Conectando ...\n");
	Client.connected = connect(Client.remote_socket, (struct sockaddr *) &Client.remote_address, sizeof(struct sockaddr_in)) != SOCKET_ERROR;
	printf("%s\n",Client.connected ?"Conectado":"falha");
}

void Client_connect() {
	Client.inicialized = 1;
	if (WSAStartup(MAKEWORD(2, 0), &Client.wsa_data)) {
		fprintf(stderr, "Falha ao criar socket\n");
	}
	Client.remote_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	memset(&Client.remote_address, 0, sizeof(struct sockaddr_in));
	Client.remote_address.sin_family = AF_INET;
	Client.remote_address.sin_addr.s_addr = inet_addr(Client.remote_ip);
	Client.remote_address.sin_port = htons(Client.remote_port);
	Client_reconnect();
}

void Client_send(void *data, size_t len, int releaseData) {
	if (!Client.connected) {
		Client_reconnect();
		if (!Client.connected) {
			fprintf(stderr, "servidor desligado\n");
			goto end;
		}
	}
	Client.connected = send(Client.remote_socket, data, len, 0) != SOCKET_ERROR;
	end:
	if (releaseData) {
		free_mem(data);
	}
}

void Client_sendTensor(Tensor tensor, int8_t figure) {
	size_t length = 0;
	void *data = tensor->serialize(tensor, &length);
	Client_send(&figure, 1, 0);
	Client_send(data, length, 1);
}

void Client_close() {
	printf("encerrando socket\n");
	WSACleanup();
	closesocket(Client.remote_socket);
}

#endif //GAB_CLIENT_H
