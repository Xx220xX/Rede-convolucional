from CNN import *
import time
from random import random

if __name__ == '__main__':
    c = CNN([20, 20, 3])
    c.addConvLayer(1,3,4)
    c.addReluLayer()
    c.addPoolLayer(1,3)
    c.addConvLayer(2,5,6)
    c.addPoolLayer(1,3)
    c.addDropOutLayer(0.5,time.time_ns())
    c.addFullConnectLayer(10, FSIGMOIG)
    c.addFullConnectLayer(1, FTANH)
    c.compile()
    inpu = [1, 0]
    out = [0]
    c.predict(inpu)
    print(c.output)
    c.save('rede0.cnn')