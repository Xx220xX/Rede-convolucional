from CNN import *
import  time
from random import  random


c = CNN([2,1,1])
c.addFullConnectLayer(5,FSIGMOIG)
c.addFullConnectLayer(5,FSIGMOIG)
c.addFullConnectLayer(1,FSIGMOIG)
c.compile()
inpu = [1,0]
out = [0]
c.predict(inpu)
print(c.output)
c.save('tester.cnn')
del c
c = None

d =CNN.load('tester.cnn')
d.predict(inpu)
print(d.output)