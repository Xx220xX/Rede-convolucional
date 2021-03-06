import logging
from threading import Thread
from CNN_GPU.CNN import *

from websocket_server import WebsocketServer

CLIENT = None


def new_client(client, server):
    print("um cliente conectou")
    # server.send_message_to_all("Hey all, a new client has joined us")


imagem = []


def receive(client, server: WebsocketServer, msg):
    global imagem
    if msg == 'imagem ppm':
        imagem = []
        return
    if msg == 'end ppm':
        c = CNN.load('redeTreinada.cnn')
        c.compile()
        c.predict(imagem)
        ans = c.output
        del c
        c = None
        print(ans)
        a = ans[0]
        k =0
        for i in range(1,len(ans)):
            if ans[i]>a:
                a = ans[i]
                k=i
        server.send_message(client, f'{k}')
        return
    imagem.append(eval(msg) / 255)




server = WebsocketServer(13254, host='127.0.0.1', loglevel=logging.INFO)
server.set_fn_new_client(new_client)
server.set_fn_message_received(receive)

server.run_forever()
