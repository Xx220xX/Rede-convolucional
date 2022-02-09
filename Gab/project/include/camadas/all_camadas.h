//
// Created by Henrique on 16/12/2021.
//

#ifndef GAB_ALL_CAMADAS_H
#define GAB_ALL_CAMADAS_H

#include"camadas/CamadaConv.h"
#include"camadas/CamadaConvF.h"
#include"camadas/CamadaConvNC.h"
#include"camadas/CamadaPool.h"
#include"camadas/CamadaRelu.h"
#include"camadas/CamadaPRelu.h"
#include"camadas/CamadaFullConnect.h"
#include"camadas/CamadaPadding.h"
#include"camadas/CamadaDropOut.h"
#include"camadas/CamadaSoftMax.h"
#include"camadas/CamadaBatchNorm.h"

#define CST_CONVOLUCAO(layer)((CamadaConv)layer)
#define Conv2D(layer)((CamadaConvF)layer)
#define CST_CONVOLUCAONC(layer)((CamadaConvNC)layer)
#define CST_POOL(layer)((CamadaPool)layer)
#define Dense(layer)((CamadaFullConnect)layer)
#define CST_PADDING(layer)((CamadaPadding )layer)
#define CST_DROPOUT(layer)((CamadaDropOut )layer)
#define CST_RELU(layer)((CamadaRelu )layer)
#define CST_PRELU(layer)((CamadaPRelu )layer)
#define CST_SOFTMAX(layer)((CamadaSoftMax)layer)
#define CST_BATCHNORM(layer)((CamadaBatchNorm)layer)
#endif //GAB_ALL_CAMADAS_H
