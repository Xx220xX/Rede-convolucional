class LIBCNN:

	def new_Tensor(self, v0, v1, v2, v3, v4, v5, v6, v7, v8): pass

	def printTensor(self, v0, v1, v2): pass

	def TensorFill(self, v0, v1, v2): pass

	def TensorFillOffSet(self, v0, v1, v2, v3): pass

	def TensorFillDouble(self, v0, v1, v2): pass

	def TensorFillDoubleOffSet(self, v0, v1, v2, v3): pass

	def TensorPutValues(self, v0, v1, v2): pass

	def TensorPutValuesOffSet(self, v0, v1, v2, v3): pass

	def TensorGetValues(self, v0, v1, v2): pass

	def TensorGetValuesOffSet(self, v0, v1, v2, v3): pass

	def TensorGetValuesMem(self, v0, v1, v2, v3): pass

	def TensorGetValuesMemOffSet(self, v0, v1, v2, v3, v4): pass

	def TensorPutValuesMem(self, v0, v1, v2, v3): pass

	def TensorPutValuesMemOffSet(self, v0, v1, v2, v3, v4): pass

	def TensorGetNorm(self, v0, v1, v2): pass

	def TensorAt(self, v0, v1, v2, v3, v4, v5): pass

	def TensorCpy(self, v0, v1, v2, v3): pass

	def TensorRandomize(self, v0, v1, v2, v3, v4): pass

	def releaseTensor(self, v0): pass

	def createCnn(self, v0, v1, v2, v3, v4): pass

	def releaseCnn(self, v0): pass

	def CnnRemoveLastLayer(self, v0): pass

	def createCnnWithWrapperFile(self, v0, v1, v2, v3, v4, v5): pass

	def createCnnWithWrapperProgram(self, v0, v1, v2, v3, v4, v5): pass

	def CnnCalculeError(self, v0, v1): pass

	def CnnCalculeErrorWithOutput(self, v0, v1, v2): pass

	def CnnCalculeErrorTWithOutput(self, v0, v1, v2): pass

	def CnnGetIndexMax(self, v0): pass

	def Convolucao(self, v0, v1, v2, v3, v4, v5, v6): pass

	def ConvolucaoNcausal(self, v0, v1, v2, v3, v4, v5, v6, v7, v8): pass

	def Pooling(self, v0, v1, v2, v3, v4): pass

	def PoolingAv(self, v0, v1, v2, v3, v4): pass

	def Relu(self, v0): pass

	def PRelu(self, v0, v1): pass

	def Padding(self, v0, v1, v2, v3, v4): pass

	def BatchNorm(self, v0, v1, v2, v3): pass

	def SoftMax(self, v0): pass

	def Dropout(self, v0, v1, v2): pass

	def FullConnect(self, v0, v1, v2, v3): pass

	def CnnCall(self, v0, v1): pass

	def CnnCallT(self, v0, v1): pass

	def CnnLearn(self, v0, v1): pass

	def CnnLearnT(self, v0, v1): pass

	def CnnInitLuaVm(self, v0): pass

	def CnnLuaConsole(self, v0): pass

	def LuaputHelpFunctionArgs(self, v0): pass

	def CnnLuaLoadString(self, v0, v1): pass

	def CnnLuaLoadFile(self, v0, v1): pass

	def cnnSave(self, v0, v1): pass

	def cnnCarregar(self, v0, v1): pass

	def normalizeGPU(self, v0, v1, v2, v3, v4, v5): pass

	def normalizeGPUSpaceKnow(self, v0, v1, v2, v3, v4, v5, v6, v7): pass

	def printCnn(self, v0): pass

	def salveCnnOutAsPPMGPU(self, v0, v1, v2): pass

	def getVersion(self): pass

	def showVersion(self): pass

	def manage2WorkDir(self, v0): pass

	def releaseManageTrain(self, v0): pass

	def manageTrainSetEvent(self, v0, v1): pass

	def manageTrainSetRun(self, v0, v1): pass

	def createManageTrain(self, v0, v1, v2, v3, v4): pass

	def ManageTrainloadImages(self, v0, v1): pass

	def ManageTraintrain(self, v0, v1): pass

	def ManageTrainfitnes(self, v0, v1): pass

	def manageTrainLoop(self, v0, v1): pass

	def createManageTrainPy(self, v0, v1, v2, v3, v4): pass

	def createManageTrainPyStr(self, v0, v1, v2, v3, v4): pass

	def PY_createCnn(self, v0, v1, v2, v3, v4, v5, v6): pass

	def PY_releaseCnn(self, v0): pass

	def CnnSaveInFile(self, v0, v1): pass

	def CnnLoadByFile(self, v0, v1): pass

	def initRandom(self, v0): pass

	def Py_getCnnOutPutAsPPM(self, v0, v1, v2, v3): pass

	def setDefaultManageThread(self): pass

	def setManageThread(self, v0, v1, v2): pass

clib:LIBCNN

