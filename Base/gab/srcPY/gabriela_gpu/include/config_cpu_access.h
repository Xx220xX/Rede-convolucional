//
// Created by Xx220xX on 04/05/2020.
//

#ifndef GAME2D_CONFIG_CPU_ACCESS_H
#define GAME2D_CONFIG_CPU_ACCESS_H
#define __kernel
#define __global
// funcionando
#define API_GPU_CALL(a, z, w, a_, b, act)\
    iter_call( a.v, a.m, a.n,z.v, z.m, z.n,w.v,w.m, w.n,a_.v, a_.m, a_.n,b.v, b.m, b.n,act)

// funcionando
#define API_GPU_LAST_LAYER(dzL, aL, out, dwL, a_L_1, w, b)\
    last_layer_learn( dzL.v,  dzL.m, aL.v, out, dwL.v, a_L_1.v,  a_L_1.m, w.v, b.v)

#define API_GPU_HIDDEN_LAYER(dzl, wl_up, dzl_up, zl, al_down, dwl, wl, bl, id_ativate_function)\
    iter_aprende(dzl.v, dzl.m,wl_up.v, wl_up.n,dzl_up.v, dzl_up.m, dzl_up.n,zl.v, dzl.n,al_down.v, \
                 al_down.m,dwl.v,wl.v,bl.v,id_ativate_function)
#define API_GPU_SET_WEIGHT(dzl, dwl, w, b, hit_learn)\
    ajusta_pesos(dzl.v, dzl.m,dwl.v, dwl.m*dwl.n ,w.v,b.v, hit_learn)
#define INIT_GPU()
#define END_GPU()
#endif //GAME2D_CONFIG_CPU_ACCESS_H
