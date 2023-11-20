#ifndef SV_CURV_COL_H
#define SV_CURV_COL_H

#include "fd_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "src_t.h"
#include <cuda_runtime.h>

/*************************************************
 * function prototype
 *************************************************/

__global__ void
sv_curv_col_el_iso_rhs_timg_z2_gpu(
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * jac3d, float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_iy, size_t siz_iz, 
    int fdx_len, int * lfdx_indx, float * lfdx_coef,
    int fdy_len, int * lfdy_indx, float * lfdy_coef,
    int fdz_len, int * lfdz_indx, float * lfdz_coef,
    const int myid, const int verbose);

__global__ void
sv_curv_col_el_iso_rhs_src_gpu(
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * jac3d, float * slw3d,
    src_t src, // short nation for reference member
    const int myid, const int verbose);

int
sv_curv_col_el_graves_Qs(float *w, int ncmp, float dt, gd_t *gd, md_t *md);

#endif
