//
// Created by hslhe on 13/11/2021.
//

#ifndef GAB_CNN_PONTO3D_H
#define GAB_CNN_PONTO3D_H
typedef struct P3d_t{
	size_t x,y,z;
}P3d ;
typedef struct P2d_t{
	size_t x,y;
}P2d ;
#define unP3D(p3d)(p3d).x,(p3d).y,(p3d).z
#define P3D(x,y,z)((P3d){x,y,z})
#define P2D(x,y)((P2d){x,y})
#endif //GAB_CNN_PONTO3D_H
