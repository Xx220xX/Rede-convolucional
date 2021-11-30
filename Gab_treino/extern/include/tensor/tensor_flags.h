//
// Created by henrique on 11/11/2021.
//

#ifndef TENSOR_TENSOR_FLAGS_H
#define TENSOR_TENSOR_FLAGS_H


/// Tensor será armazenado na ram
#define TENSOR_RAM 0b00100
/// Tensor será armazenado na memoria compartilhada
#define TENSOR_SVM 0b01000
/// Tensor será armazenado no dispositivo(default)
#define TENSOR_GPU 0b00000

/// Será feita a copia do ponteiro passado
#define TENSOR_CPY 0b10000

/// Tensor 3D (defautl).
#define TENSOR3D 0b0
/// Tensor 4D.
#define TENSOR4D 0b00000001

/// Tensor tipo REAL (default).
#define TENSOR_REAL 0x0
/// Tensor tipo char.
#define TENSOR_CHAR 0b00100000
/// Tensor tipo int.
#define TENSOR_INT 0b01000000

#define TENSOR_MASK_DIM         0x00000001
#define TENSOR_MASK_MEM         0b00001100
#define TENSOR_MASK_CPY         0b00010000
#define TENSOR_MASK_TYPE        0b01100000

#endif //TENSOR_TENSOR_FLAGS_H
