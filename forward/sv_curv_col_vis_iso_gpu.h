#ifndef SV_CURV_COL_VIS_ISO_H
#define SV_CURV_COL_VIS_ISO_H

#include "fd_t.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "src_t.h"
#include "bdry_t.h"
#include "io_funcs.h"
#include <cuda_runtime.h>

/*************************************************
 * function prototype
 *************************************************/
int
sv_curv_col_vis_iso_onestage(
  float *w_cur,
  float *rhs, 
  wav_t  wav_d,
  fd_wav_t fd_wav_d,
  gd_t   gd_d,
  gd_metric_t  metric_d,
  md_t md_d,
  bdrypml_t  bdrypml_d,
  bdryfree_t  bdryfree_d,
  src_t src_d,
  // include different order/stentil
  int num_of_fdx_op, fd_op_t *fdx_op,
  int num_of_fdy_op, fd_op_t *fdy_op,
  int num_of_fdz_op, fd_op_t *fdz_op,
  int fdz_max_len, 
  const int myid, const int verbose);

__global__ void
sv_curv_col_vis_iso_atten_gpu(
    float *w_cur,
    float *rhs, 
    wav_t  wav_d,
    md_t md_d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_iy, size_t siz_iz,
    const int myid, const int verbose);

__global__ void
sv_curv_col_vis_iso_free_gpu(float *w_end,
                             wav_t  wav_d,
                             gd_t   gd_d,
                             gd_metric_t  metric_d,
                             md_t md_d,
                             bdryfree_t  bdryfree_d,
                             const int myid, 
                             const int verbose);

int
sv_curv_col_vis_iso_dvh2dvz(gd_t            *gd,
                            gd_metric_t     *metric,
                            md_t            *md,
                            bdryfree_t      *bdryfree,
                            int fd_len,
                            int *fd_indx,
                            float *fd_coef,
                            const int verbose);

#endif
