# A fast API to deep learn
VERSION documentation 2.3
### The class DNN
 ##### construtor
 ```python
    DNN(self, arquitetura, taxaAprendizado=0.1):
```
return a instance for use to deep learn
arquitetura is architeture of network must be tuple or list 
taxaAprendizado is a hit learn paramter , by default is 0.1

#### sample to learn xor operation

````python

import DNN
input = [[1,1],[1,0],[0,1],[0,0]]
target = [[int(i^j)] for i,j in input]
dnn = DNN((2,6,5,3,1))
epocas  = 1000
error = 1e-3
for ep  in range(epocas):
    e = 0
    for i in range(len(inp)):
        dnn(inp[i])
        dnn.aprende(target[i])
        e += (dnn.out[0] - target[i][0])**2
    e /= 2
    if e<error:
        print('after ',ep,'epics i learn it') 
        break;

````

