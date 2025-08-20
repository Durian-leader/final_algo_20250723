#ifndef _MFCC_DATA_H_
#define _MFCC_DATA_H_ 

#include "arm_math_types.h"


#ifdef   __cplusplus
extern "C"
{
#endif


/*****

 DCT COEFFICIENTS FOR THE MFCC

*****/


#define NB_MFCC_DCT_COEFS_DCT13 520
extern const float32_t mfcc_dct_coefs_dct13[NB_MFCC_DCT_COEFS_DCT13];



/*****

 WINDOW COEFFICIENTS

*****/


#define NB_MFCC_WIN_COEFS_HANN256 256
extern const float32_t mfcc_window_coefs_hann256[NB_MFCC_WIN_COEFS_HANN256];



/*****

 MEL FILTER COEFFICIENTS FOR THE MFCC

*****/

#define NB_MFCC_NB_FILTER_MEL40 40
extern const uint32_t mfcc_filter_pos_mel40[NB_MFCC_NB_FILTER_MEL40];
extern const uint32_t mfcc_filter_len_mel40[NB_MFCC_NB_FILTER_MEL40];





#define NB_MFCC_FILTER_COEFS_MEL40 247
extern const float32_t mfcc_filter_coefs_mel40[NB_MFCC_FILTER_COEFS_MEL40];



#ifdef   __cplusplus
}
#endif

#endif

