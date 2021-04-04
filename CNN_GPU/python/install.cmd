cd C:\Users\Henrique\Desktop\CNN\CNN_GPU\python
copy "C:\Users\Henrique\Desktop\CNN\CNN_GPU\cmake-build-debug\libCNNGPU.dll"  "C:\Users\Henrique\Desktop\CNN\CNN_GPU\python\CNN_GPU\lib\libCNNGPU.dll"
python setup.py install

rmdir /S /Q build 
pause