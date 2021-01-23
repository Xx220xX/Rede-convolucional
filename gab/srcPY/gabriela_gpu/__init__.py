from gabriela_gpu.dnn_gpu import DNN,ALAN,TANH,RELU,SIGMOID,SOFTMAX,IDENTIFY
from gabriela_gpu.dnn_gpu import setSeed
# for retro compatible version
RND = DNN

if __name__ == '__main__':  # teste
    dnn = DNN((1,3,5,4,1))
    setSeed(1)
    print(dnn([1]))
    dnn.learn([0])
    print(dnn([1]))
    setSeed(1)
    dnn.randomize()
    print(dnn([1]))
