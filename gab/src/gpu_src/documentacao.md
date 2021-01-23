Segue abaixo a lista de todas as funçoes usadas pelo kernel
## Comuns

| Nome  | IDENTIFICADOR |argumentos |
| :---  | :---       | :---       |
| gab_find_last_dzl | KERNEL_FIND_LAST_DZL | a[L], y, dz[L], b[L], hit learn, index beguin |
| gab_find_dwl | KERNEL_FIND_LAST_DZL | dz[l], a[l-1], a[l-1].m, dw[l], index beguin |
| gab_find_dwl_and_update | KERNEL_FIND_AND_UPDATE_DWL | w[l], dz[l], a[l-1], a[l-1].m, dw[l], hit learn, index beguin |
| gab_sum | KERNEL_SUM | ans, x , n |

## Não normalizado
###### Propagação
| Nome  | IDENTIFICADOR |argumentos |
| :---  | :---       | :---       |
| gab_feed_alan    | KERNEL_FULL_FEED +  ALAN    | w[l] ,a[l-1], a[l-1].m , b[l] , z[l], a[l], index begin |
| gab_feed_tanh    | KERNEL_FULL_FEED +  TANH    | w[l] ,a[l-1], a[l-1].m , b[l] , z[l], a[l], index begin |
| gab_feed_relu    | KERNEL_FULL_FEED +  RELU    | w[l] ,a[l-1], a[l-1].m , b[l] , z[l], a[l], index begin |
| gab_feed_sigmoid | KERNEL_FULL_FEED +  SIGMOID | w[l] ,a[l-1], a[l-1].m , b[l] , z[l], a[l], index begin |
| gab_feed_softmax | KERNEL_FULL_FEED +  SOFTMAX | w[l] ,a[l-1], a[l-1].m , b[l] , z[l], a[l], index begin |

###### Retropropagação
| Nome  | IDENTIFICADOR |argumentos |
| :---  | :---       | :---       |
| gab_find_dzl_alan     | KERNEL_FIND_DZL + ALAN     | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin | 
| gab_find_dzl_tanh     | KERNEL_FIND_DZL + TANH     | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin | 
| gab_find_dzl_relu     | KERNEL_FIND_DZL + RELU     | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin | 
| gab_find_dzl_sigmoid  | KERNEL_FIND_DZL + SIGMOID  | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin | 
| gab_find_dzl_softmax  | KERNEL_FIND_DZL + SOFTMAX  | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin |
| gab_find_dzl_identify | KERNEL_FIND_DZL + IDENTIFY | dw[l+1], w[l+1], z[l+1],z[l+1].m, z[l],a[l],dz[l],b[l], hit learn, index begin |


## Normalizado
###### Propagação

| Nome  | IDENTIFICADOR |argumentos |
| :---  | :---       | :---       |
| gab_feed | KERNEL_WA_B_FEED | w[l],  a[l-1], a[l-1].m,b[l],z[l], index begin |
| gab_normalize_alan | KERNEL_NORMALIZE +  ALAN | a[l] ,z[l], media, desvio padrao , index begin |
| gab_normalize_tanh | KERNEL_NORMALIZE +  TANH | a[l] ,z[l], media, desvio padrao , index begin |
| gab_normalize_relu | KERNEL_NORMALIZE +  RELU | a[l] ,z[l], media, desvio padrao , index begin |
| gab_normalize_sigmoid | KERNEL_NORMALIZE +  SIGMOID | a[l] ,z[l], media, desvio padrao , index begin |
| gab_normalize_softmax | KERNEL_NORMALIZE +  SOFTMAX | a[l] ,z[l], media, desvio padrao , index begin |
| gab_normalize_identify | KERNEL_NORMALIZE +  IDENTIFY | a[l] ,z[l], media, desvio padrao , index begin |
| gab_sum | KERNEL_SUM | ans, x, n |
| gab_desvio_padrao | KERNEL_STD | ans, x, n |
| gab_divide_vector | KERNEL_STD | quociente, dividendo, divisor |

###### Retropropagação
| Nome  | IDENTIFICADOR |argumentos |
| :---  | :---       | :---       |
| gab_norm_find_dzl_alan | KERNEL_FIND_DZL_NORMALIZE + ALAN         | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 
| gab_norm_find_dzl_tanh | KERNEL_FIND_DZL_NORMALIZE + TANH         | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 
| gab_norm_find_dzl_relu | KERNEL_FIND_DZL_NORMALIZE + RELU         | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 
| gab_norm_find_dzl_sigmoid | KERNEL_FIND_DZL_NORMALIZE + SIGMOID   | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 
| gab_norm_find_dzl_softmax | KERNEL_FIND_DZL_NORMALIZE + SOFTMAX   | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 
| gab_norm_find_dzl_identify | KERNEL_FIND_DZL_NORMALIZE + IDENTIFY | dw[l+1], w[l+1], dz[l+1], dz[l+1].m, z[l], a[l],dz[l],b[l], hit learn,media, desvio, n, index begin | 



