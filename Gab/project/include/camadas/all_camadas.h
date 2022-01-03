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

#define CST_CONVOLUCAO(cnn, layer)((CamadaConv)cnn->cm[layer])
#define CST_CONVOLUCAOF(cnn, layer)((CamadaConvF)cnn->cm[layer])
#define CST_CONVOLUCAO2D(cnn, layer)((CamadaConv2D)cnn->cm[layer])
#define CST_CONVOLUCAONC(cnn, layer)((CamadaConvNC)cnn->cm[layer])
#define CST_POOL(cnn, layer)((CamadaPool)cnn->cm[layer])
#define CST_FULLCONNECT(cnn, layer)((CamadaFullConnect)cnn->cm[layer])
#define CST_PADDING(cnn, layer)((CamadaPadding )cnn->cm[layer])
#define CST_DROPOUT(cnn, layer)((CamadaDropOut )cnn->cm[layer])
#define CST_RELU(cnn, layer)((CamadaRelu )cnn->cm[layer])
#define CST_PRELU(cnn, layer)((CamadaPRelu )cnn->cm[layer])
#define CST_SOFTMAX(cnn, layer)((CamadaSoftMax)cnn->cm[layer])
#define CST_BATCHNORM(cnn, layer)((CamadaBatchNorm)cnn->cm[layer])
#endif //GAB_ALL_CAMADAS_H
