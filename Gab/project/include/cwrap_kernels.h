#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedMacroInspection"
//utils.h
//bathnorm.h
#define Knew_BatchNormMedia(x) KRN_new(x, "BatchNormMedia", 5, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormMedia(kname, kernel_iter_Len, Vr_a, Vw_u, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_a, &Vw_u, &int_ax, &int_ay)

#define Knew_BatchNormInvDesv(x) KRN_new(x, "BatchNormInvDesv", 7, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormInvDesv(kname, kernel_iter_Len, Vr_a, Vr_u, Vr_o, REAL_episolon, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_a, &Vr_u, &Vr_o, &REAL_episolon, &int_ax, &int_ay)

#define Knew_BatchNormNormaliza(x) KRN_new(x, "BatchNormNormaliza", 10, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormNormaliza(kname, kernel_iter_Len, Vw_s, Vrw_v, Vr_a, Vr_u, Vr_o, Vr_Y, Vr_B, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vw_s, &Vrw_v, &Vr_a, &Vr_u, &Vr_o, &Vr_Y, &Vr_B, &int_ax, &int_ay)

#define Knew_BatchNormaCalcDnorm(x) KRN_new(x, "BatchNormaCalcDnorm", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormaCalcDnorm(kname, kernel_iter_Len, Vw_dv, Vr_ds, Vr_Y, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vw_dv, &Vr_ds, &Vr_Y, &int_ax, &int_ay)

#define Knew_BatchNormMediadnorm_norma(x) KRN_new(x, "BatchNormMediadnorm_norma", 7, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormMediadnorm_norma(kname, kernel_iter_Len, Vr_v, Vr_dv, Vr_mdnorm, Vr_mdnormnorm, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_v, &Vr_dv, &Vr_mdnorm, &Vr_mdnormnorm, &int_ax, &int_ay)

#define Knew_BatchNormaCalcDa(x) KRN_new(x, "BatchNormaCalcDa", 9, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormaCalcDa(kname, kernel_iter_Len, Vr_da, Vr_v, Vr_dv, Vr_mdnorm, Vr_mdnormnorm, Vr_o, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_da, &Vr_v, &Vr_dv, &Vr_mdnorm, &Vr_mdnormnorm, &Vr_o, &int_ax, &int_ay)

#define Knew_BatchNormaCalcdYdB(x) KRN_new(x, "BatchNormaCalcdYdB", 8, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_BatchNormaCalcdYdB(kname, kernel_iter_Len, Vr_ds, Vr_v, Vw_dY, Vw_dB, long_batchSize, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_v, &Vw_dY, &Vw_dB, &long_batchSize, &int_ax, &int_ay)

#define Knew_BatchNormaLearn(x) KRN_new(x, "BatchNormaLearn", 8, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_BatchNormaLearn(kname, kernel_iter_Len, Vrw_Y, Vrw_B, Vrw_dY, Vrw_dB, REAL_hit, REAL_momento, REAL_decaimento) Execute(kname, kernel_iter_Len, &Vrw_Y, &Vrw_B, &Vrw_dY, &Vrw_dB, &REAL_hit, &REAL_momento, &REAL_decaimento)

//cnnutils.h
#define Knew_createImg(x) KRN_new(x, "createImg", 7, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_createImg(kname, kernel_iter_Len, __global_unsigned_char__out, Vr_v, int_vx, int_vy, int_imi, int_imy) Execute(kname, kernel_iter_Len, &__global_unsigned_char__out, &Vr_v, &int_vx, &int_vy, &int_imi, &int_imy)

#define Knew_putIMG(x) KRN_new(x, "putIMG", 12, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_putIMG(kname, kernel_iter_Len, __global_unsigned_char__imagem_saida, Vr_v, int_z, REAL_px, REAL_py, int_imy, int_width, int_i0, int_j0, int_vx, int_vy) Execute(kname, kernel_iter_Len, &__global_unsigned_char__imagem_saida, &Vr_v, &int_z, &REAL_px, &REAL_py, &int_imy, &int_width, &int_i0, &int_j0, &int_vx, &int_vy)

#define Knew_normalizeVector(x) KRN_new(x, "normalizeVector", 6, sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_normalizeVector(kname, kernel_iter_Len, Vr_input, Vr_saida, REAL_multiplicador, REAL_somador, REAL_subtrator) Execute(kname, kernel_iter_Len, &Vr_input, &Vr_saida, &REAL_multiplicador, &REAL_somador, &REAL_subtrator)

#define Knew_kernel_sub(x) KRN_new(x, "kernel_sub", 4, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int))
#define KExec_kernel_sub(kname, kernel_iter_Len, Vr_ds, Vr_s, Vr_t) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_s, &Vr_t)

#define Knew_kernel_normalizechar2real(x) KRN_new(x, "kernel_normalizechar2real", 5, sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_kernel_normalizechar2real(kname, kernel_iter_Len, Vr_dst, __global_unsigned_char__src, REAL_a, REAL_b) Execute(kname, kernel_iter_Len, &Vr_dst, &__global_unsigned_char__src, &REAL_a, &REAL_b)

#define Knew_kernel_getVetorClassFromChar(x) KRN_new(x, "kernel_getVetorClassFromChar", 4, sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_int))
#define KExec_kernel_getVetorClassFromChar(kname, kernel_iter_Len, Vr_dst, __global_unsigned_char__ints, unsigned_int_noptiobs) Execute(kname, kernel_iter_Len, &Vr_dst, &__global_unsigned_char__ints, &unsigned_int_noptiobs)

#define Knew_kernel_fixW(x) KRN_new(x, "kernel_fixW", 6, sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_kernel_fixW(kname, kernel_iter_Len, Vr_w, Vr_dw, REAL_hitlearn, REAL_momento, REAL_decaimentoDePeso) Execute(kname, kernel_iter_Len, &Vr_w, &Vr_dw, &REAL_hitlearn, &REAL_momento, &REAL_decaimentoDePeso)

//conv.h
#define Knew_convSum(x) KRN_new(x, "convSum", 13, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convSum(kname, kernel_iter_Len, Vr_filtro, Vr_entrada, Vr_saida, int_passox, int_passoy, int_saidatx, int_saidaty, int_entradatx, int_entradaty, int_fx, int_fy, int_fz) Execute(kname, kernel_iter_Len, &Vr_filtro, &Vr_entrada, &Vr_saida, &int_passox, &int_passoy, &int_saidatx, &int_saidaty, &int_entradatx, &int_entradaty, &int_fx, &int_fy, &int_fz)

#define Knew_convCalcGradAndFixWeight(x) KRN_new(x, "convCalcGradAndFixWeight", 17, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_convCalcGradAndFixWeight(kname, kernel_iter_Len, Vr_filtros, Vr_ds, Vr_entrada, Vr_gradFiltro, int_fx, int_fy, int_fz, int_entrada_tx, int_entrada_ty, int_saida_tx, int_saida_ty, int_passox, int_passoy, REAL_hitLearn, REAL_momento, REAL_weightDecay) Execute(kname, kernel_iter_Len, &Vr_filtros, &Vr_ds, &Vr_entrada, &Vr_gradFiltro, &int_fx, &int_fy, &int_fz, &int_entrada_tx, &int_entrada_ty, &int_saida_tx, &int_saida_ty, &int_passox, &int_passoy, &REAL_hitLearn, &REAL_momento, &REAL_weightDecay)

#define Knew_convCalcGradIn(x) KRN_new(x, "convCalcGradIn", 14, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convCalcGradIn(kname, kernel_iter_Len, Vr_filtro, Vr_gradEntrada, Vr_gradNext, int_fx, int_fy, int_fz, int_passox, int_passoy, int_entradatx, int_entradaty, int_saidatx, int_saidaty, int_saidatz) Execute(kname, kernel_iter_Len, &Vr_filtro, &Vr_gradEntrada, &Vr_gradNext, &int_fx, &int_fy, &int_fz, &int_passox, &int_passoy, &int_entradatx, &int_entradaty, &int_saidatx, &int_saidaty, &int_saidatz)

#define Knew_convCalcGradBatch(x) KRN_new(x, "convCalcGradBatch", 14, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convCalcGradBatch(kname, kernel_iter_Len, Vr_ds, Vr_A, Vr_dW, long_batchSize, int_fx, int_fy, int_fz, int_entrada_tx, int_entrada_ty, int_saida_tx, int_saida_ty, int_passox, int_passoy) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_A, &Vr_dW, &long_batchSize, &int_fx, &int_fy, &int_fz, &int_entrada_tx, &int_entrada_ty, &int_saida_tx, &int_saida_ty, &int_passox, &int_passoy)

//conv2d.h
#define Knew_conv2dSum(x) KRN_new(x, "conv2dSum", 16, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_conv2dSum(kname, kernel_iter_Len, Vr_W, Vr_a, Vw_Z, Vw_s, int_px, int_py, int_sx, int_sy, int_ax, int_ay, int_az, int_fx, int_fy, int_fz, int_fid) Execute(kname, kernel_iter_Len, &Vr_W, &Vr_a, &Vw_Z, &Vw_s, &int_px, &int_py, &int_sx, &int_sy, &int_ax, &int_ay, &int_az, &int_fx, &int_fy, &int_fz, &int_fid)

#define Knew_conv2dCalcGradZ(x) KRN_new(x, "conv2dCalcGradZ", 5, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int))
#define KExec_conv2dCalcGradZ(kname, kernel_iter_Len, Vr_ds, Vr_z, Vw_dz, int_fid) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_z, &Vw_dz, &int_fid)

#define Knew_conv2dCalcGradIn(x) KRN_new(x, "conv2dCalcGradIn", 14, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_conv2dCalcGradIn(kname, kernel_iter_Len, Vr_W, Vw_da, Vr_dz, int_fx, int_fy, int_fz, int_px, int_py, int_atx, int_aty, int_az, int_sx, int_sy) Execute(kname, kernel_iter_Len, &Vr_W, &Vw_da, &Vr_dz, &int_fx, &int_fy, &int_fz, &int_px, &int_py, &int_atx, &int_aty, &int_az, &int_sx, &int_sy)

#define Knew_conv2dCalcGradAndFixWeight(x) KRN_new(x, "conv2dCalcGradAndFixWeight", 17, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_conv2dCalcGradAndFixWeight(kname, kernel_iter_Len, Vrw_W, Vr_dz, Vr_a, Vrw_dW, int_fx, int_fy, int_ax, int_ay, int_az, int_sx, int_sy, int_px, int_py, REAL_hitLearn, REAL_momento, REAL_weightDecay) Execute(kname, kernel_iter_Len, &Vrw_W, &Vr_dz, &Vr_a, &Vrw_dW, &int_fx, &int_fy, &int_ax, &int_ay, &int_az, &int_sx, &int_sy, &int_px, &int_py, &REAL_hitLearn, &REAL_momento, &REAL_weightDecay)

#define Knew_conv2dCalcGradBatch(x) KRN_new(x, "conv2dCalcGradBatch", 15, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_conv2dCalcGradBatch(kname, kernel_iter_Len, Vr_dz, Vr_a, Vr_dW, long_batchSize, int_fx, int_fy, int_fz, int_ax, int_ay, int_az, int_sx, int_sy, int_px, int_py) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_a, &Vr_dW, &long_batchSize, &int_fx, &int_fy, &int_fz, &int_ax, &int_ay, &int_az, &int_sx, &int_sy, &int_px, &int_py)

//convf.h
#define Knew_convFSum(x) KRN_new(x, "convFSum", 16, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convFSum(kname, kernel_iter_Len, Vr_W, Vr_B, Vr_A, Vw_Z, Vw_S, int_px, int_py, int_sx, int_sy, int_atx, int_aty, int_fx, int_fy, int_fz, int_fid) Execute(kname, kernel_iter_Len, &Vr_W, &Vr_B, &Vr_A, &Vw_Z, &Vw_S, &int_px, &int_py, &int_sx, &int_sy, &int_atx, &int_aty, &int_fx, &int_fy, &int_fz, &int_fid)

#define Knew_convFCalcGradZ(x) KRN_new(x, "convFCalcGradZ", 5, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int))
#define KExec_convFCalcGradZ(kname, kernel_iter_Len, Vr_ds, Vr_z, Vw_dz, int_fid) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_z, &Vw_dz, &int_fid)

#define Knew_convFCalcGradBAndFix(x) KRN_new(x, "convFCalcGradBAndFix", 9, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_convFCalcGradBAndFix(kname, kernel_iter_Len, Vrw_B, Vrw_dB, Vr_dZ, int_dzx, int_dzy, REAL_hitLearn, REAL_momento, REAL_weightDecay) Execute(kname, kernel_iter_Len, &Vrw_B, &Vrw_dB, &Vr_dZ, &int_dzx, &int_dzy, &REAL_hitLearn, &REAL_momento, &REAL_weightDecay)

#define Knew_convFCalcGradBBatch(x) KRN_new(x, "convFCalcGradBBatch", 6, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_long), sizeof(cl_int))
#define KExec_convFCalcGradBBatch(kname, kernel_iter_Len, Vrw_dB, Vr_dZ, int_dzx, int_dzy, long_batchSize) Execute(kname, kernel_iter_Len, &Vrw_dB, &Vr_dZ, &int_dzx, &int_dzy, &long_batchSize)

#define Knew_convFCalcGradIn(x) KRN_new(x, "convFCalcGradIn", 14, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convFCalcGradIn(kname, kernel_iter_Len, Vr_W, Vw_da, Vr_dz, int_fx, int_fy, int_fz, int_px, int_py, int_atx, int_aty, int_sx, int_sy, int_sz) Execute(kname, kernel_iter_Len, &Vr_W, &Vw_da, &Vr_dz, &int_fx, &int_fy, &int_fz, &int_px, &int_py, &int_atx, &int_aty, &int_sx, &int_sy, &int_sz)

#define Knew_convFCalcGradAndFixWeight(x) KRN_new(x, "convFCalcGradAndFixWeight", 17, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_convFCalcGradAndFixWeight(kname, kernel_iter_Len, Vr_W, Vr_dz, Vr_a, Vr_dW, int_fx, int_fy, int_fz, int_a_tx, int_a_ty, int_s_tx, int_s_ty, int_px, int_py, REAL_hitLearn, REAL_momento, REAL_weightDecay) Execute(kname, kernel_iter_Len, &Vr_W, &Vr_dz, &Vr_a, &Vr_dW, &int_fx, &int_fy, &int_fz, &int_a_tx, &int_a_ty, &int_s_tx, &int_s_ty, &int_px, &int_py, &REAL_hitLearn, &REAL_momento, &REAL_weightDecay)

#define Knew_convFCalcGradBatch(x) KRN_new(x, "convFCalcGradBatch", 14, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_convFCalcGradBatch(kname, kernel_iter_Len, Vr_dz, Vr_a, Vr_dW, long_batchSize, int_fx, int_fy, int_fz, int_a_tx, int_a_ty, int_s_tx, int_s_ty, int_px, int_py) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_a, &Vr_dW, &long_batchSize, &int_fx, &int_fy, &int_fz, &int_a_tx, &int_a_ty, &int_s_tx, &int_s_ty, &int_px, &int_py)

//convNc.h
#define Knew_convncSum(x) KRN_new(x, "convncSum", 17, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int))
#define KExec_convncSum(kname, kernel_iter_Len, Vr_W, Vr_A, Vr_Z, Vr_S, unsigned_int_fid, unsigned_int_passox, int_passoy, unsigned_int_largx, unsigned_int_largy, unsigned_int_entradatx, unsigned_int_entradaty, unsigned_int_saidatx, unsigned_int_saidaty, unsigned_int_fx, unsigned_int_fy, unsigned_int_fz) Execute(kname, kernel_iter_Len, &Vr_W, &Vr_A, &Vr_Z, &Vr_S, &unsigned_int_fid, &unsigned_int_passox, &int_passoy, &unsigned_int_largx, &unsigned_int_largy, &unsigned_int_entradatx, &unsigned_int_entradaty, &unsigned_int_saidatx, &unsigned_int_saidaty, &unsigned_int_fx, &unsigned_int_fy, &unsigned_int_fz)

#define Knew_convncCalcGradZ(x) KRN_new(x, "convncCalcGradZ", 5, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_int))
#define KExec_convncCalcGradZ(kname, kernel_iter_Len, Vr_ds, Vr_z, Vr_dz, unsigned_int_fid) Execute(kname, kernel_iter_Len, &Vr_ds, &Vr_z, &Vr_dz, &unsigned_int_fid)

#define Knew_convncCalcGrads(x) KRN_new(x, "convncCalcGrads", 15, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int))
#define KExec_convncCalcGrads(kname, kernel_iter_Len, Vr_W, Vr_DA, Vr_dz, unsigned_int_passox, unsigned_int_passoy, unsigned_int_largx, unsigned_int_largy, unsigned_int_entradatx, unsigned_int_entradaty, unsigned_int_saidatx, unsigned_int_saidaty, unsigned_int_fx, unsigned_int_fy, unsigned_int_fz) Execute(kname, kernel_iter_Len, &Vr_W, &Vr_DA, &Vr_dz, &unsigned_int_passox, &unsigned_int_passoy, &unsigned_int_largx, &unsigned_int_largy, &unsigned_int_entradatx, &unsigned_int_entradaty, &unsigned_int_saidatx, &unsigned_int_saidaty, &unsigned_int_fx, &unsigned_int_fy, &unsigned_int_fz)

#define Knew_convncCalcFiltro(x) KRN_new(x, "convncCalcFiltro", 19, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_convncCalcFiltro(kname, kernel_iter_Len, Vr_dz, Vr_A, Vr_W, Vr_dW, unsigned_int_dw_x, unsigned_int_dw_y, unsigned_int_dw_z, unsigned_int_a_x, unsigned_int_a_y, unsigned_int_s_x, unsigned_int_s_y, unsigned_int_passox, unsigned_int_passoy, unsigned_int_largx, unsigned_int_largy, REAL_hitlearn, REAL_momento, REAL_weightDecay) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_A, &Vr_W, &Vr_dW, &unsigned_int_dw_x, &unsigned_int_dw_y, &unsigned_int_dw_z, &unsigned_int_a_x, &unsigned_int_a_y, &unsigned_int_s_x, &unsigned_int_s_y, &unsigned_int_passox, &unsigned_int_passoy, &unsigned_int_largx, &unsigned_int_largy, &REAL_hitlearn, &REAL_momento, &REAL_weightDecay)

#define Knew_convncCalcFiltroBatch(x) KRN_new(x, "convncCalcFiltroBatch", 16, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int))
#define KExec_convncCalcFiltroBatch(kname, kernel_iter_Len, Vr_dz, Vr_A, Vr_dW, long_batchSize, unsigned_int_dw_x, unsigned_int_dw_y, unsigned_int_dw_z, unsigned_int_a_x, unsigned_int_a_y, unsigned_int_s_x, unsigned_int_s_y, unsigned_int_passox, unsigned_int_passoy, unsigned_int_largx, unsigned_int_largy) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_A, &Vr_dW, &long_batchSize, &unsigned_int_dw_x, &unsigned_int_dw_y, &unsigned_int_dw_z, &unsigned_int_a_x, &unsigned_int_a_y, &unsigned_int_s_x, &unsigned_int_s_y, &unsigned_int_passox, &unsigned_int_passoy, &unsigned_int_largx, &unsigned_int_largy)

//dropout.h
#define Knew_dropativa(x) KRN_new(x, "dropativa", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_dropativa(kname, kernel_iter_Len, Vr_entrada, Vr_saida, __global_char__hitmap, long_seed, REAL_pativa) Execute(kname, kernel_iter_Len, &Vr_entrada, &Vr_saida, &__global_char__hitmap, &long_seed, &REAL_pativa)

#define Knew_dropcalcgrad(x) KRN_new(x, "dropcalcgrad", 4, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int))
#define KExec_dropcalcgrad(kname, kernel_iter_Len, Vr_gradentrada, __global_char__hitmap, Vr_gradnext) Execute(kname, kernel_iter_Len, &Vr_gradentrada, &__global_char__hitmap, &Vr_gradnext)

//fullconnect.h
#define Knew_fullfeed(x) KRN_new(x, "fullfeed", 9, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_fullfeed(kname, kernel_iter_Len, Vr_a, Vr_w, Vr_b, Vr_z, Vr_s, int_fid, int_w_x, int_w_y) Execute(kname, kernel_iter_Len, &Vr_a, &Vr_w, &Vr_b, &Vr_z, &Vr_s, &int_fid, &int_w_x, &int_w_y)

#define Knew_fullCalcDWandFix(x) KRN_new(x, "fullCalcDWandFix", 9, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int), sizeof(cl_int))
#define KExec_fullCalcDWandFix(kname, kernel_iter_Len, Vr_a, Vr_w, Vr_dw, Vr_dz, REAL_hitlearn, REAL_momento, REAL_decaimentoDePeso, int_pesosy) Execute(kname, kernel_iter_Len, &Vr_a, &Vr_w, &Vr_dw, &Vr_dz, &REAL_hitlearn, &REAL_momento, &REAL_decaimentoDePeso, &int_pesosy)

#define Knew_fullCalcDz(x) KRN_new(x, "fullCalcDz", 5, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int))
#define KExec_fullCalcDz(kname, kernel_iter_Len, Vr_dz, Vr_ds, Vr_z, int_dfa) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_ds, &Vr_z, &int_dfa)

#define Knew_fullCalcDzBath(x) KRN_new(x, "fullCalcDzBath", 7, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_long), sizeof(cl_int))
#define KExec_fullCalcDzBath(kname, kernel_iter_Len, Vr_dz, Vr_ds, Vr_z, Vr_db, int_dfa, long_batchSize) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_ds, &Vr_z, &Vr_db, &int_dfa, &long_batchSize)

#define Knew_fullCalcDzAndFixB(x) KRN_new(x, "fullCalcDzAndFixB", 10, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_fullCalcDzAndFixB(kname, kernel_iter_Len, Vr_dz, Vr_ds, Vr_z, Vr_b, Vr_db, int_dfa, REAL_hitlearn, REAL_momento, REAL_decaimentoDePeso) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_ds, &Vr_z, &Vr_b, &Vr_db, &int_dfa, &REAL_hitlearn, &REAL_momento, &REAL_decaimentoDePeso)

#define Knew_fullcalcin(x) KRN_new(x, "fullcalcin", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_fullcalcin(kname, kernel_iter_Len, Vr_dz, Vr_da, Vr_w, int_pesosx, int_pesosy) Execute(kname, kernel_iter_Len, &Vr_dz, &Vr_da, &Vr_w, &int_pesosx, &int_pesosy)

#define Knew_fullCalcDWBatch(x) KRN_new(x, "fullCalcDWBatch", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int), sizeof(cl_int))
#define KExec_fullCalcDWBatch(kname, kernel_iter_Len, Vr_a, Vr_dw, Vr_dz, long_batchSize, int_pesosy) Execute(kname, kernel_iter_Len, &Vr_a, &Vr_dw, &Vr_dz, &long_batchSize, &int_pesosy)

//padding.h
#define Knew_paddingfeed(x) KRN_new(x, "paddingfeed", 9, sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int))
#define KExec_paddingfeed(kname, kernel_iter_Len, Vr_in, Vr_out, unsigned_int_txi, unsigned_int_tyi, unsigned_int_txo, unsigned_int_tyo, unsigned_int_t, unsigned_int_l) Execute(kname, kernel_iter_Len, &Vr_in, &Vr_out, &unsigned_int_txi, &unsigned_int_tyi, &unsigned_int_txo, &unsigned_int_tyo, &unsigned_int_t, &unsigned_int_l)

#define Knew_paddingBack(x) KRN_new(x, "paddingBack", 9, sizeof(void *), sizeof(void *), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_uint), sizeof(cl_int))
#define KExec_paddingBack(kname, kernel_iter_Len, Vr_gradNext, Vr_gradin, unsigned_int_txi, unsigned_int_tyi, unsigned_int_txo, unsigned_int_tyo, unsigned_int_t, unsigned_int_l) Execute(kname, kernel_iter_Len, &Vr_gradNext, &Vr_gradin, &unsigned_int_txi, &unsigned_int_tyi, &unsigned_int_txo, &unsigned_int_tyo, &unsigned_int_t, &unsigned_int_l)

//poolav.h
#define Knew_poolAVativa(x) KRN_new(x, "poolAVativa", 11, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_poolAVativa(kname, kernel_iter_Len, Vr_entrada, Vr_saida, int_passox, int_passoy, int_fx, int_fy, int_saidatx, int_saidaty, int_entradatx, int_entradaty) Execute(kname, kernel_iter_Len, &Vr_entrada, &Vr_saida, &int_passox, &int_passoy, &int_fx, &int_fy, &int_saidatx, &int_saidaty, &int_entradatx, &int_entradaty)

#define Knew_poolAvCalcGrads(x) KRN_new(x, "poolAvCalcGrads", 13, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_poolAvCalcGrads(kname, kernel_iter_Len, Vr_A, Vw_dA, Vr_dS, Vr_S, int_fx, int_fy, int_px, int_py, int_entradatx, int_entradaty, int_saidatx, int_saidaty) Execute(kname, kernel_iter_Len, &Vr_A, &Vw_dA, &Vr_dS, &Vr_S, &int_fx, &int_fy, &int_px, &int_py, &int_entradatx, &int_entradaty, &int_saidatx, &int_saidaty)

//poolMax.h
#define Knew_poolativa(x) KRN_new(x, "poolativa", 11, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_poolativa(kname, kernel_iter_Len, Vr_entrada, Vr_saida, int_passox, int_passoy, int_filtrox, int_filtroy, int_saidatx, int_saidaty, int_entradatx, int_entradaty) Execute(kname, kernel_iter_Len, &Vr_entrada, &Vr_saida, &int_passox, &int_passoy, &int_filtrox, &int_filtroy, &int_saidatx, &int_saidaty, &int_entradatx, &int_entradaty)

#define Knew_poolCalcGrads(x) KRN_new(x, "poolCalcGrads", 13, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_poolCalcGrads(kname, kernel_iter_Len, Vr_A, Vr_dA, Vr_dS, Vr_S, int_fx, int_fy, int_px, int_py, int_entradatx, int_entradaty, int_saidatx, int_saidaty) Execute(kname, kernel_iter_Len, &Vr_A, &Vr_dA, &Vr_dS, &Vr_S, &int_fx, &int_fy, &int_px, &int_py, &int_entradatx, &int_entradaty, &int_saidatx, &int_saidaty)

//poolMin.h
#define Knew_poolativaMin(x) KRN_new(x, "poolativaMin", 11, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_poolativaMin(kname, kernel_iter_Len, Vr_A, Vr_S, int_passox, int_passoy, int_filtrox, int_filtroy, int_saidatx, int_saidaty, int_entradatx, int_entradaty) Execute(kname, kernel_iter_Len, &Vr_A, &Vr_S, &int_passox, &int_passoy, &int_filtrox, &int_filtroy, &int_saidatx, &int_saidaty, &int_entradatx, &int_entradaty)

//prelu.h
#define Knew_preluativa(x) KRN_new(x, "preluativa", 4, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int))
#define KExec_preluativa(kname, kernel_iter_Len, Vr_A, Vw_S, Vr_W) Execute(kname, kernel_iter_Len, &Vr_A, &Vw_S, &Vr_W)

#define Knew_prelucalcgrad(x) KRN_new(x, "prelucalcgrad", 10, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_prelucalcgrad(kname, kernel_iter_Len, Vw_dA, Vr_A, Vr_dS, Vrw_W, Vrw_dW, int_learn, REAL_hitlearn, REAL_momento, REAL_decaimento) Execute(kname, kernel_iter_Len, &Vw_dA, &Vr_A, &Vr_dS, &Vrw_W, &Vrw_dW, &int_learn, &REAL_hitlearn, &REAL_momento, &REAL_decaimento)

#define Knew_preluonlyfix(x) KRN_new(x, "preluonlyfix", 8, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_preluonlyfix(kname, kernel_iter_Len, Vr_A, Vr_dS, Vrw_W, Vrw_dW, REAL_hitlearn, REAL_momento, REAL_decaimento) Execute(kname, kernel_iter_Len, &Vr_A, &Vr_dS, &Vrw_W, &Vrw_dW, &REAL_hitlearn, &REAL_momento, &REAL_decaimento)

#define Knew_prelucalcgradBatch(x) KRN_new(x, "prelucalcgradBatch", 7, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int))
#define KExec_prelucalcgradBatch(kname, kernel_iter_Len, Vw_dA, Vr_A, Vr_dS, Vr_W, Vrw_dW, long_batchSize) Execute(kname, kernel_iter_Len, &Vw_dA, &Vr_A, &Vr_dS, &Vr_W, &Vrw_dW, &long_batchSize)

#define Knew_preluonlyDABatch(x) KRN_new(x, "preluonlyDABatch", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_long), sizeof(cl_int))
#define KExec_preluonlyDABatch(kname, kernel_iter_Len, Vr_A, Vr_dS, Vr_W, Vr_dW, long_batchSize) Execute(kname, kernel_iter_Len, &Vr_A, &Vr_dS, &Vr_W, &Vr_dW, &long_batchSize)

//relu.h
#define Knew_reluativa(x) KRN_new(x, "reluativa", 5, sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_reluativa(kname, kernel_iter_Len, Vr_A, Vr_S, REAL_menor, REAL_maior) Execute(kname, kernel_iter_Len, &Vr_A, &Vr_S, &REAL_menor, &REAL_maior)

#define Knew_relucalcgrad(x) KRN_new(x, "relucalcgrad", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(CL_REAL), sizeof(CL_REAL), sizeof(cl_int))
#define KExec_relucalcgrad(kname, kernel_iter_Len, Vr_dA, Vr_A, Vr_dS, REAL_menor, REAL_maior) Execute(kname, kernel_iter_Len, &Vr_dA, &Vr_A, &Vr_dS, &REAL_menor, &REAL_maior)

//softmax.h
#define Knew_softmaxExp(x) KRN_new(x, "softmaxExp", 3, sizeof(void *), sizeof(void *), sizeof(cl_int))
#define KExec_softmaxExp(kname, kernel_iter_Len, Vr_entrada, Vr_exponent) Execute(kname, kernel_iter_Len, &Vr_entrada, &Vr_exponent)

#define Knew_softmaxSomaExp(x) KRN_new(x, "softmaxSomaExp", 5, sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_softmaxSomaExp(kname, kernel_iter_Len, Vr_eps, Vr_soma, int_saidatx, int_saidaty) Execute(kname, kernel_iter_Len, &Vr_eps, &Vr_soma, &int_saidatx, &int_saidaty)

#define Knew_softmaxNormaliza(x) KRN_new(x, "softmaxNormaliza", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_softmaxNormaliza(kname, kernel_iter_Len, Vr_exponet, Vr_soma, Vr_saida, int_saidatx, int_saidaty) Execute(kname, kernel_iter_Len, &Vr_exponet, &Vr_soma, &Vr_saida, &int_saidatx, &int_saidaty)

#define Knew_softMaxcalcgrad(x) KRN_new(x, "softMaxcalcgrad", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_softMaxcalcgrad(kname, kernel_iter_Len, Vr_da, Vr_s, Vr_ds, int_sx, int_sy) Execute(kname, kernel_iter_Len, &Vr_da, &Vr_s, &Vr_ds, &int_sx, &int_sy)

#define Knew_softmaxFindMax(x) KRN_new(x, "softmaxFindMax", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_softmaxFindMax(kname, kernel_iter_Len, Vr_a, Vr_mx, __global_int__i_max, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_a, &Vr_mx, &__global_int__i_max, &int_ax, &int_ay)

#define Knew_softmaxExpNorm(x) KRN_new(x, "softmaxExpNorm", 6, sizeof(void *), sizeof(void *), sizeof(void *), sizeof(cl_int), sizeof(cl_int), sizeof(cl_int))
#define KExec_softmaxExpNorm(kname, kernel_iter_Len, Vr_entrada, Vr_exponent, Vr_mx, int_ax, int_ay) Execute(kname, kernel_iter_Len, &Vr_entrada, &Vr_exponent, &Vr_mx, &int_ax, &int_ay)

#pragma clang diagnostic pop
