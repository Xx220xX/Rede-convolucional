//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_PONTO3D_H
#define GAB_CNN_PONTO3D_H
typedef struct P3d_t{
	size_t x,y,z;
}P3d ;
typedef struct P2d_t{
	size_t x,y,z;
}P2d ;
#define P3D(x,...)((Ponto3d){x,##__VA_ARGS__})
#endif //GAB_CNN_PONTO3D_H
