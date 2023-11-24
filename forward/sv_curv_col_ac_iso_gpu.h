#ifndef SV_CURV_COL_AC_ISO_H
#define SV_CURV_COL_AC_ISO_H

#include "fd_t.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "src_t.h"
#include "bdry_t.h"
#include "io_funcs.h"

/*************************************************
 * function prototype
 *************************************************/

int
sv_curv_col_ac_iso_onestage(
  float *w_cur_d,
  float *rhs_d, 
  wav_t  wav_d,
  fd_wav_t fd_wav_d,
  gd_t  gd_d,
  gd_metric_t metric_d,
  md_t md_d,
  bdrypml_t  bdrypml_d,
  bdryfree_t bdryfree_d,
  src_t src_d,
  // include different order/stentil
  int num_of_fdx_op, fd_op_t *fdx_op,
  int num_of_fdy_op, fd_op_t *fdy_op,
  int num_of_fdz_op, fd_op_t *fdz_op,
  int fdz_max_len, 
  const int myid, const int verbose);

__global__ void
sv_curv_col_ac_iso_rhs_inner_gpu(
    float *Vx, float *Vy, float *Vz, float *P, 
    float *hVx, float *hVy, float *hVz, float *hP, 
    float *xi_x, float *xi_y, float *xi_z,
    float *et_x, float *et_y, float *et_z,
    float *zt_x, float *zt_y, float *zt_z,
    float *kappa3d, float *slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_iy, size_t siz_iz,
    size_t *lfdx_shift, float *lfdx_coef,
    size_t *lfdy_shift, float *lfdy_coef,
    size_t *lfdz_shift, float *lfdz_coef,
    int myid, int verbose);

__global__ void
sv_curv_col_ac_iso_rhs_timg_z2_gpu(
                   float *P, int ni, int nj, int ni1,  
                   int nj1, int nk2, int nz, 
                   size_t siz_iy, size_t siz_iz,
                   int myid, int verbose);

__global__ void
sv_curv_col_ac_iso_rhs_vlow_z2_gpu(
    float *Vx , float *Vy , float *Vz ,
    float *hP, 
    float *xi_x, float *xi_y, float *xi_z,
    float *et_x, float *et_y, float *et_z,
    float *zt_x, float *zt_y, float *zt_z,
    float *kappa3d, float *slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_iy, size_t siz_iz,
    int fdx_len, size_t *lfdx_shift, float *lfdx_coef,
    int fdy_len, size_t *lfdy_shift, float *lfdy_coef,
    int num_of_fdz_op, int fdz_max_len, int *fdz_len,
    float *lfdz_coef_all, size_t *lfdz_shift_all,
    int myid, int verbose);

int
sv_curv_col_ac_iso_rhs_cfspml(
    float *Vx, float *Vy, float *Vz, float *P, 
    float *hVx , float *hVy , float *hVz, float *hP,
    float *xi_x, float *xi_y, float *xi_z,
    float *et_x, float *et_y, float *et_z,
    float *zt_x, float *zt_y, float *zt_z,
    float *kappa3d, float *slw3d,
    size_t siz_iy, size_t siz_iz,
    size_t *lfdx_shift, float *lfdx_coef,
    size_t *lfdy_shift, float *lfdy_coef,
    size_t *lfdz_shift, float *lfdz_coef,
    bdrypml_t bdrypml_d,
    int myid, int verbose);

__global__ void
sv_curv_col_ac_iso_rhs_cfspml_gpu(int idim, int iside,
                                  float *Vx, float *Vy, float *Vz, float *P,
                                  float *hVx, float *hVy, float *hVz, float *hP,
                                  float *xi_x, float *xi_y, float *xi_z,
                                  float *et_x, float *et_y, float *et_z,
                                  float *zt_x, float *zt_y, float *zt_z,
                                  float *kappa3d, float *slw3d,
                                  size_t siz_iy, size_t siz_iz,
                                  size_t *lfdx_shift, float *lfdx_coef,
                                  size_t *lfdy_shift, float *lfdy_coef,
                                  size_t *lfdz_shift, float *lfdz_coef,
                                  bdrypml_t bdrypml_d,
                                  int myid, int verbose);

__global__ void
sv_curv_col_ac_iso_rhs_src_gpu(
    float *hVx , float *hVy , float *hVz ,
    float *hP, 
    float *jac3d, float *slw3d,
    src_t src, 
    int myid, int verbose);

#endif
