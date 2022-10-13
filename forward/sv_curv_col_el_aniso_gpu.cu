/*******************************************************************************
 * solver of isotropic elastic 1st-order eqn using curv grid and macdrp schem
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "sv_curv_col_el_gpu.h"
#include "sv_curv_col_el_aniso_gpu.h"
#include "cuda_common.h"

//#define SV_EQ1ST_CURV_COLGRD_ISO_DEBUG

/*******************************************************************************
 * perform one stage calculation of rhs
 ******************************************************************************/

void
sv_curv_col_el_aniso_onestage(
  float *w_cur_d,
  float *rhs_d, 
  wav_t  wav_d,
  fd_wav_t fd_wav_d,
  gdinfo_t  gdinfo_d,
  gdcurv_metric_t metric_d,
  md_t md_d,
  bdrypml_t  bdrypml_d,
  bdryfree_t bdryfree_d,
  src_t src_d,
  // include different order/stentil
  int num_of_fdx_op, fd_op_t *fdx_op,
  int num_of_fdy_op, fd_op_t *fdy_op,
  int num_of_fdz_op, fd_op_t *fdz_op,
  int fdz_max_len, 
  const int myid, const int verbose)
{
  // local pointer get each vars
  float *Vx    = w_cur_d + wav_d.Vx_pos ;
  float *Vy    = w_cur_d + wav_d.Vy_pos ;
  float *Vz    = w_cur_d + wav_d.Vz_pos ;
  float *Txx   = w_cur_d + wav_d.Txx_pos;
  float *Tyy   = w_cur_d + wav_d.Tyy_pos;
  float *Tzz   = w_cur_d + wav_d.Tzz_pos;
  float *Txz   = w_cur_d + wav_d.Txz_pos;
  float *Tyz   = w_cur_d + wav_d.Tyz_pos;
  float *Txy   = w_cur_d + wav_d.Txy_pos;
  float *hVx   = rhs_d   + wav_d.Vx_pos ; 
  float *hVy   = rhs_d   + wav_d.Vy_pos ; 
  float *hVz   = rhs_d   + wav_d.Vz_pos ; 
  float *hTxx  = rhs_d   + wav_d.Txx_pos; 
  float *hTyy  = rhs_d   + wav_d.Tyy_pos; 
  float *hTzz  = rhs_d   + wav_d.Tzz_pos; 
  float *hTxz  = rhs_d   + wav_d.Txz_pos; 
  float *hTyz  = rhs_d   + wav_d.Tyz_pos; 
  float *hTxy  = rhs_d   + wav_d.Txy_pos; 

  float *xi_x  = metric_d.xi_x;
  float *xi_y  = metric_d.xi_y;
  float *xi_z  = metric_d.xi_z;
  float *et_x  = metric_d.eta_x;
  float *et_y  = metric_d.eta_y;
  float *et_z  = metric_d.eta_z;
  float *zt_x  = metric_d.zeta_x;
  float *zt_y  = metric_d.zeta_y;
  float *zt_z  = metric_d.zeta_z;
  float *jac3d = metric_d.jac;

  float *c11   = md_d.c11;
  float *c12   = md_d.c12;
  float *c13   = md_d.c13;
  float *c14   = md_d.c14;
  float *c15   = md_d.c15;
  float *c16   = md_d.c16;
  float *c22   = md_d.c22;
  float *c23   = md_d.c23;
  float *c24   = md_d.c24;
  float *c25   = md_d.c25;
  float *c26   = md_d.c26;
  float *c33   = md_d.c33;
  float *c34   = md_d.c34;
  float *c35   = md_d.c35;
  float *c36   = md_d.c36;
  float *c44   = md_d.c44;
  float *c45   = md_d.c45;
  float *c46   = md_d.c46;
  float *c55   = md_d.c55;
  float *c56   = md_d.c56;
  float *c66   = md_d.c66;
  float *slw3d = md_d.rho;

  // grid size
  int ni1 = gdinfo_d.ni1;
  int ni2 = gdinfo_d.ni2;
  int nj1 = gdinfo_d.nj1;
  int nj2 = gdinfo_d.nj2;
  int nk1 = gdinfo_d.nk1;
  int nk2 = gdinfo_d.nk2;

  int ni  = gdinfo_d.ni;
  int nj  = gdinfo_d.nj;
  int nk  = gdinfo_d.nk;
  int nx  = gdinfo_d.nx;
  int ny  = gdinfo_d.ny;
  int nz  = gdinfo_d.nz;
  size_t siz_iy   = gdinfo_d.siz_iy;
  size_t siz_iz   = gdinfo_d.siz_iz;
  size_t siz_icmp = gdinfo_d.siz_icmp;

  float *matVx2Vz = bdryfree_d.matVx2Vz2;
  float *matVy2Vz = bdryfree_d.matVy2Vz2;

  // local fd op
  int    fdx_len;
  int    *fdx_indx;
  float  *fdx_coef;
  int    fdy_len;
  int    *fdy_indx;
  float  *fdy_coef;
  int    fdz_len;
  int    *fdz_indx;
  float  *fdz_coef;

  // for get a op from 1d array, currently use num_of_fdz_op as index
  // length, index, coef of a op
  fdx_len  = fdx_op[num_of_fdx_op-1].total_len;
  fdx_indx = fdx_op[num_of_fdx_op-1].indx;
  fdx_coef = fdx_op[num_of_fdx_op-1].coef;

  fdy_len  = fdy_op[num_of_fdy_op-1].total_len;
  fdy_indx = fdy_op[num_of_fdy_op-1].indx;
  fdy_coef = fdy_op[num_of_fdy_op-1].coef;

  fdz_len  = fdz_op[num_of_fdz_op-1].total_len;
  fdz_indx = fdz_op[num_of_fdz_op-1].indx;
  fdz_coef = fdz_op[num_of_fdz_op-1].coef;

  // use local stack array for speedup
  float  lfdx_coef [fdx_len];
  size_t lfdx_shift[fdx_len];
  float  lfdy_coef [fdy_len];
  size_t lfdy_shift[fdy_len];
  float  lfdz_coef [fdz_len];
  size_t lfdz_shift[fdz_len];

  // put fd op into local array
  for (int i=0; i < fdx_len; i++) {
    lfdx_coef [i] = fdx_coef[i];
    lfdx_shift[i] = fdx_indx[i];
  }
  for (int j=0; j < fdy_len; j++) {
    lfdy_coef [j] = fdy_coef[j];
    lfdy_shift[j] = fdy_indx[j] * siz_iy;
  }
  for (int k=0; k < fdz_len; k++) {
    lfdz_coef [k] = fdz_coef[k];
    lfdz_shift[k] = fdz_indx[k] * siz_iz;
  }

  // allocate max_len because fdz may have different lens
  // these array is for low order surface
  float  fdz_coef_all [num_of_fdz_op*fdz_max_len];
  size_t fdz_shift_all[num_of_fdz_op*fdz_max_len];
  int    fdz_len_all[num_of_fdz_op];
  // loop near surface layers
  for (int n=0; n < num_of_fdz_op-1; n++)
  {
    // get pos and len for this point
    fdz_len_all[n]  = fdz_op[n].total_len;
    // point to indx/coef for this point
    int   *p_fdz_indx  = fdz_op[n].indx;
    float *p_fdz_coef  = fdz_op[n].coef;
    for (int n_fd = 0; n_fd < fdz_len_all[n] ; n_fd++) {
      fdz_shift_all[n_fd + n*fdz_max_len]  = p_fdz_indx[n_fd] * siz_iz;
      fdz_coef_all [n_fd + n*fdz_max_len]  = p_fdz_coef[n_fd];
    }
  }

  int  *lfdz_len_d = fd_wav_d.fdz_len_d;
  float *lfdx_coef_d = fd_wav_d.fdx_coef_d;
  float *lfdy_coef_d = fd_wav_d.fdy_coef_d;
  float *lfdz_coef_d = fd_wav_d.fdz_coef_d;
  float *lfdz_coef_all_d = fd_wav_d.fdz_coef_all_d;
  size_t  *lfdx_shift_d = fd_wav_d.fdx_shift_d;
  size_t  *lfdy_shift_d = fd_wav_d.fdy_shift_d;
  size_t  *lfdz_shift_d = fd_wav_d.fdz_shift_d;
  size_t  *lfdz_shift_all_d = fd_wav_d.fdz_shift_all_d;
  int  *lfdx_indx_d = fd_wav_d.fdx_indx_d;
  int  *lfdy_indx_d = fd_wav_d.fdy_indx_d;
  int  *lfdz_indx_d = fd_wav_d.fdz_indx_d;
  int  *lfdz_indx_all_d = fd_wav_d.fdz_indx_all_d;
  //host to device
  CUDACHECK(cudaMemcpy(lfdx_coef_d,lfdx_coef,fdx_len*sizeof(float),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdy_coef_d,lfdy_coef,fdy_len*sizeof(float),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_coef_d,lfdz_coef,fdz_len*sizeof(float),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdx_shift_d,lfdx_shift,fdx_len*sizeof(size_t),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdy_shift_d,lfdy_shift,fdy_len*sizeof(size_t),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_shift_d,lfdz_shift,fdz_len*sizeof(size_t),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdx_indx_d,fdx_indx,fdx_len*sizeof(int),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdy_indx_d,fdy_indx,fdy_len*sizeof(int),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_indx_d,fdz_indx,fdz_len*sizeof(int),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_len_d,fdz_len_all,num_of_fdz_op*sizeof(int),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_coef_all_d,fdz_coef_all,fdz_max_len*num_of_fdz_op*sizeof(float),cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(lfdz_shift_all_d,fdz_shift_all,fdz_max_len*num_of_fdz_op*sizeof(size_t),cudaMemcpyHostToDevice));

  {
    dim3 block(8,8,8);
    dim3 grid;
    grid.x = (ni+block.x-1)/block.x;
    grid.y = (nj+block.y-1)/block.y;
    grid.z = (nk+block.z-1)/block.z;
    sv_curv_col_el_aniso_rhs_inner_gpu <<<grid, block>>> (
                        Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                        hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        c11,c12,c13,c14,c15,c16,
                            c22,c23,c24,c25,c26,
                                c33,c34,c35,c36,
                                    c44,c45,c46,
                                        c55,c56,
                                            c66, slw3d,
                        ni1,ni,nj1,nj,nk1,nk,siz_iy,siz_iz,
                        lfdx_shift_d, lfdx_coef_d,
                        lfdy_shift_d, lfdy_coef_d,
                        lfdz_shift_d, lfdz_coef_d,
                        myid, verbose);
  }

  // free surface at z2
  if (bdryfree_d.is_sides_free[2][1] == 1)
  {
    // tractiong
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_el_iso_rhs_timg_z2_gpu  <<<grid, block>>> (
                          Txx,Tyy,Tzz,Txz,Tyz,Txy,hVx,hVy,hVz,
                          xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                          jac3d, slw3d,
                          ni1,ni,nj1,nj,nk1,nk2,siz_iy,siz_iz,
                          fdx_len, lfdx_indx_d, lfdx_coef_d,
                          fdy_len, lfdy_indx_d, lfdy_coef_d,
                          fdz_len, lfdz_indx_d, lfdz_coef_d,
                          myid, verbose);
      cudaDeviceSynchronize();
    }

    // velocity: vlow
    {
      dim3 block(8,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_el_aniso_rhs_vlow_z2_gpu  <<<grid, block>>> (
                        Vx,Vy,Vz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        c11,c12,c13,c14,c15,c16,
                            c22,c23,c24,c25,c26,
                                c33,c34,c35,c36,
                                    c44,c45,c46,
                                        c55,c56,
                                            c66, slw3d,
                        matVx2Vz,matVy2Vz,
                        ni1,ni,nj1,nj,nk1,nk2,siz_iy,siz_iz,
                        fdx_len, lfdx_shift_d, lfdx_coef_d,
                        fdy_len, lfdy_shift_d, lfdy_coef_d,
                        num_of_fdz_op,fdz_max_len,lfdz_len_d,
                        lfdz_coef_all_d,lfdz_shift_all_d,
                        myid, verbose);
      CUDACHECK( cudaDeviceSynchronize() );
    }
  }

  // cfs-pml, loop face inside
  if (bdrypml_d.is_enable_pml == 1)
  {
    sv_curv_col_el_aniso_rhs_cfspml(Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                                    hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                                    xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                    c11,c12,c13,c14,c15,c16,
                                        c22,c23,c24,c25,c26,
                                            c33,c34,c35,c36,
                                                c44,c45,c46,
                                                    c55,c56,
                                                        c66, slw3d,
                                    nk2, siz_iy,siz_iz,
                                    lfdx_shift_d, lfdx_coef_d,
                                    lfdy_shift_d, lfdy_coef_d,
                                    lfdz_shift_d, lfdz_coef_d,
                                    bdrypml_d, bdryfree_d,
                                    myid, verbose);
  }

  // add source term
  if (src_d.total_number > 0)
  {
    {
      dim3 block(256);
      dim3 grid;
      grid.x = (src_d.total_number+block.x-1) / block.x;
      sv_curv_col_el_iso_rhs_src_gpu  <<< grid,block >>> (
                        hVx, hVy, hVz, hTxx, hTyy, hTzz, hTxz, hTyz, hTxy,
                        jac3d, slw3d, 
                        src_d,
                        myid, verbose);
      CUDACHECK( cudaDeviceSynchronize() );
    }
  }

  return;
}

/*******************************************************************************
 * calculate all points without boundaries treatment
 ******************************************************************************/

__global__ void
sv_curv_col_el_aniso_rhs_inner_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * c11d, float * c12d, float * c13d,
    float * c14d, float * c15d, float * c16d,
                  float * c22d, float * c23d,
    float * c24d, float * c25d, float * c26d,
                                float * c33d,
    float * c34d, float * c35d, float * c36d,
    float * c44d, float * c45d, float * c46d,
                  float * c55d, float * c56d,
                                float * c66d,
                                float * slw3d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_iy, size_t siz_iz,
    size_t * lfdx_shift, float * lfdx_coef,
    size_t * lfdy_shift, float * lfdy_coef,
    size_t * lfdz_shift, float * lfdz_coef,
    const int myid, const int verbose)
{
  // use local stack array for speedup
  // local var
  float DxTxx,DxTyy,DxTzz,DxTxy,DxTxz,DxTyz,DxVx,DxVy,DxVz;
  float DyTxx,DyTyy,DyTzz,DyTxy,DyTxz,DyTyz,DyVx,DyVy,DyVz;
  float DzTxx,DzTyy,DzTzz,DzTxy,DzTxz,DzTyz,DzVx,DzVy,DzVz;
  float slw;
  float c11,c12,c13,c14,c15,c16;
  float     c22,c23,c24,c25,c26;
  float         c33,c34,c35,c36;
  float             c44,c45,c46;
  float                 c55,c56;
  float                     c66;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  float * Vx_ptr;
  float * Vy_ptr;
  float * Vz_ptr;
  float * Txx_ptr;
  float * Txy_ptr;
  float * Txz_ptr;
  float * Tyy_ptr;
  float * Tzz_ptr;
  float * Tyz_ptr;


  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  // caclu all points
  if(ix<ni && iy<nj && iz<nk)
  {
    size_t iptr = (ix+ni1) + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;

    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    Txx_ptr = Txx + iptr;
    Tyy_ptr = Tyy + iptr;
    Tzz_ptr = Tzz + iptr;
    Txz_ptr = Txz + iptr;
    Tyz_ptr = Tyz + iptr;
    Txy_ptr = Txy + iptr;

    // Vx derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVx, Vx_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyVx, Vx_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzVx, Vx_ptr, lfdz_shift, lfdz_coef);

    // Vy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVy, Vy_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyVy, Vy_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzVy, Vy_ptr, lfdz_shift, lfdz_coef);

    // Vz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxVz, Vz_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyVz, Vz_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzVz, Vz_ptr, lfdz_shift, lfdz_coef);

    // Txx derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxx, Txx_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTxx, Txx_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTxx, Txx_ptr, lfdz_shift, lfdz_coef);

    // Tyy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTyy, Tyy_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTyy, Tyy_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTyy, Tyy_ptr, lfdz_shift, lfdz_coef);

    // Tzz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTzz, Tzz_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTzz, Tzz_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTzz, Tzz_ptr, lfdz_shift, lfdz_coef);

    // Txz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxz, Txz_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTxz, Txz_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTxz, Txz_ptr, lfdz_shift, lfdz_coef);

    // Tyz derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTyz, Tyz_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTyz, Tyz_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTyz, Tyz_ptr, lfdz_shift, lfdz_coef);

    // Txy derivatives
    M_FD_SHIFT_PTR_MACDRP(DxTxy, Txy_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyTxy, Txy_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzTxy, Txy_ptr, lfdz_shift, lfdz_coef);

    // metric
    xix = xi_x[iptr];
    xiy = xi_y[iptr];
    xiz = xi_z[iptr];
    etx = et_x[iptr];
    ety = et_y[iptr];
    etz = et_z[iptr];
    ztx = zt_x[iptr];
    zty = zt_y[iptr];
    ztz = zt_z[iptr];

    // medium
    slw = slw3d[iptr];
    c11 = c11d[iptr];
    c12 = c12d[iptr];
    c13 = c13d[iptr];
    c14 = c14d[iptr];
    c15 = c15d[iptr];
    c16 = c16d[iptr];
    c22 = c22d[iptr];
    c23 = c23d[iptr];
    c24 = c24d[iptr];
    c25 = c25d[iptr];
    c26 = c26d[iptr];
    c33 = c33d[iptr];
    c34 = c34d[iptr];
    c35 = c35d[iptr];
    c36 = c36d[iptr];
    c44 = c44d[iptr];
    c45 = c45d[iptr];
    c46 = c46d[iptr];
    c55 = c55d[iptr];
    c56 = c56d[iptr];
    c66 = c66d[iptr];

    // moment equation
    hVx[iptr] = slw*( xix*DxTxx + xiy*DxTxy + xiz*DxTxz  
                     +etx*DyTxx + ety*DyTxy + etz*DyTxz 
                     +ztx*DzTxx + zty*DzTxy + ztz*DzTxz );
    hVy[iptr] = slw*( xix*DxTxy + xiy*DxTyy + xiz*DxTyz
                     +etx*DyTxy + ety*DyTyy + etz*DyTyz
                     +ztx*DzTxy + zty*DzTyy + ztz*DzTyz );
    hVz[iptr] = slw*( xix*DxTxz + xiy*DxTyz + xiz*DxTzz 
                     +etx*DyTxz + ety*DyTyz + etz*DyTzz
                     +ztx*DzTxz + zty*DzTyz + ztz*DzTzz );

    // Hooke's equatoin

	  hTxx[iptr] = (c11*xix + c16*xiy + c15*xiz) * DxVx + (c16*xix + c12*xiy + c14*xiz) * DxVy + (c15*xix + c14*xiy + c13*xiz) * DxVz
               + (c11*etx + c16*ety + c15*etz) * DyVx + (c16*etx + c12*ety + c14*etz) * DyVy + (c15*etx + c14*ety + c13*etz) * DyVz
               + (c11*ztx + c16*zty + c15*ztz) * DzVx + (c16*ztx + c12*zty + c14*ztz) * DzVy + (c15*ztx + c14*zty + c13*ztz) * DzVz;
    
    hTyy[iptr] = (c12*xix + c26*xiy + c25*xiz) * DxVx + (c26*xix + c22*xiy + c24*xiz) * DxVy + (c25*xix + c24*xiy + c23*xiz) * DxVz
               + (c12*etx + c26*ety + c25*etz) * DyVx + (c26*etx + c22*ety + c24*etz) * DyVy + (c25*etx + c24*ety + c23*etz) * DyVz
               + (c12*ztx + c26*zty + c25*ztz) * DzVx + (c26*ztx + c22*zty + c24*ztz) * DzVy + (c25*ztx + c24*zty + c23*ztz) * DzVz;
    
    hTzz[iptr] = (c13*xix + c36*xiy + c35*xiz) * DxVx + (c36*xix + c23*xiy + c34*xiz) * DxVy + (c35*xix + c34*xiy + c33*xiz) * DxVz
               + (c13*etx + c36*ety + c35*etz) * DyVx + (c36*etx + c23*ety + c34*etz) * DyVy + (c35*etx + c34*ety + c33*etz) * DyVz
               + (c13*ztx + c36*zty + c35*ztz) * DzVx + (c36*ztx + c23*zty + c34*ztz) * DzVy + (c35*ztx + c34*zty + c33*ztz) * DzVz;
  

    hTyz[iptr] = (c14*xix + c46*xiy + c45*xiz) * DxVx + (c46*xix + c24*xiy + c44*xiz) * DxVy + (c45*xix + c44*xiy + c34*xiz) * DxVz
               + (c14*etx + c46*ety + c45*etz) * DyVx + (c46*etx + c24*ety + c44*etz) * DyVy + (c45*etx + c44*ety + c34*etz) * DyVz
               + (c14*ztx + c46*zty + c45*ztz) * DzVx + (c46*ztx + c24*zty + c44*ztz) * DzVy + (c45*ztx + c44*zty + c34*ztz) * DzVz;
  
    hTxz[iptr] = (c15*xix + c56*xiy + c55*xiz) * DxVx + (c56*xix + c25*xiy + c45*xiz) * DxVy + (c55*xix + c45*xiy + c35*xiz) * DxVz
               + (c15*etx + c56*ety + c55*etz) * DyVx + (c56*etx + c25*ety + c45*etz) * DyVy + (c55*etx + c45*ety + c35*etz) * DyVz
               + (c15*ztx + c56*zty + c55*ztz) * DzVx + (c56*ztx + c25*zty + c45*ztz) * DzVy + (c55*ztx + c45*zty + c35*ztz) * DzVz;
  
    hTxy[iptr] = (c16*xix + c66*xiy + c56*xiz) * DxVx + (c66*xix + c26*xiy + c46*xiz) * DxVy + (c56*xix + c46*xiy + c36*xiz) * DxVz
               + (c16*etx + c66*ety + c56*etz) * DyVx + (c66*etx + c26*ety + c46*etz) * DyVy + (c56*etx + c46*ety + c36*etz) * DyVz
               + (c16*ztx + c66*zty + c56*ztz) * DzVx + (c66*ztx + c26*zty + c46*ztz) * DzVy + (c56*ztx + c46*zty + c36*ztz) * DzVz;
  }

  return;
}

/*******************************************************************************
 * free surface boundary
 ******************************************************************************/

/*
 * implement vlow boundary
 */

__global__ void
sv_curv_col_el_aniso_rhs_vlow_z2_gpu(
    float *  Vx , float *  Vy , float *  Vz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * c11d, float * c12d, float * c13d,
    float * c14d, float * c15d, float * c16d,
                  float * c22d, float * c23d,
    float * c24d, float * c25d, float * c26d,
                                float * c33d,
    float * c34d, float * c35d, float * c36d,
    float * c44d, float * c45d, float * c46d,
                  float * c55d, float * c56d,
                                float * c66d,
                                float * slw3d,
    float * matVx2Vz, float * matVy2Vz,
    int ni1, int ni, int nj1, int nj, int nk1, int nk2,
    size_t siz_iy, size_t siz_iz,
    int fdx_len, size_t * lfdx_shift, float * lfdx_coef,
    int fdy_len, size_t * lfdy_shift, float * lfdy_coef,
    int num_of_fdz_op, int fdz_max_len, int * fdz_len,
    float *lfdz_coef_all, size_t *lfdz_shift_all,
    const int myid, const int verbose)
{
  // local var
  int k;
  int n_fd; // loop var for fd
  int lfdz_len;
  // local var
  float DxVx,DxVy,DxVz;
  float DyVx,DyVy,DyVz;
  float DzVx,DzVy,DzVz;
  float slw;
  float c11,c12,c13,c14,c15,c16;
  float     c22,c23,c24,c25,c26;
  float         c33,c34,c35,c36;
  float             c44,c45,c46;
  float                 c55,c56;
  float                     c66;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  float lfdz_coef[5] = {0.0};
  int   lfdz_shift[5] = {0};
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  // loop near surface layers
  for (int n=0; n < num_of_fdz_op-1; n++)
  {
    // conver to k index, from surface to inner
    k = nk2 - n;
    // get pos and len for this point
    lfdz_len  = fdz_len[n];
    for (n_fd = 0; n_fd < lfdz_len ; n_fd++) {
      lfdz_shift[n_fd] = lfdz_shift_all[n*fdz_max_len+n_fd];
      lfdz_coef [n_fd]  = lfdz_coef_all [n*fdz_max_len+n_fd];
    }
    if(ix<ni && iy<nj)
    {
      size_t iptr   = (ix+ni1) + (iy+nj1) * siz_iy + k * siz_iz;

      // metric
      xix = xi_x[iptr];
      xiy = xi_y[iptr];
      xiz = xi_z[iptr];
      etx = et_x[iptr];
      ety = et_y[iptr];
      etz = et_z[iptr];
      ztx = zt_x[iptr];
      zty = zt_y[iptr];
      ztz = zt_z[iptr];

      // medium
      slw = slw3d[iptr];
      c11 = c11d[iptr];
      c12 = c12d[iptr];
      c13 = c13d[iptr];
      c14 = c14d[iptr];
      c15 = c15d[iptr];
      c16 = c16d[iptr];
      c22 = c22d[iptr];
      c23 = c23d[iptr];
      c24 = c24d[iptr];
      c25 = c25d[iptr];
      c26 = c26d[iptr];
      c33 = c33d[iptr];
      c34 = c34d[iptr];
      c35 = c35d[iptr];
      c36 = c36d[iptr];
      c44 = c44d[iptr];
      c45 = c45d[iptr];
      c46 = c46d[iptr];
      c55 = c55d[iptr];
      c56 = c56d[iptr];
      c66 = c66d[iptr];

      // Vx derivatives
      M_FD_SHIFT(DxVx, Vx, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVx, Vx, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      // Vy derivatives
      M_FD_SHIFT(DxVy, Vy, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVy, Vy, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      // Vz derivatives
      M_FD_SHIFT(DxVz, Vz, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVz, Vz, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      if (k==nk2) // at surface, convert
      {
        size_t ij = ((ix+ni1) + (iy+nj1) * siz_iy)*9;
        DzVx = matVx2Vz[ij+3*0+0] * DxVx
             + matVx2Vz[ij+3*0+1] * DxVy
             + matVx2Vz[ij+3*0+2] * DxVz
             + matVy2Vz[ij+3*0+0] * DyVx
             + matVy2Vz[ij+3*0+1] * DyVy
             + matVy2Vz[ij+3*0+2] * DyVz;

        DzVy = matVx2Vz[ij+3*1+0] * DxVx
             + matVx2Vz[ij+3*1+1] * DxVy
             + matVx2Vz[ij+3*1+2] * DxVz
             + matVy2Vz[ij+3*1+0] * DyVx
             + matVy2Vz[ij+3*1+1] * DyVy
             + matVy2Vz[ij+3*1+2] * DyVz;

        DzVz = matVx2Vz[ij+3*2+0] * DxVx
             + matVx2Vz[ij+3*2+1] * DxVy
             + matVx2Vz[ij+3*2+2] * DxVz
             + matVy2Vz[ij+3*2+0] * DyVx
             + matVy2Vz[ij+3*2+1] * DyVy
             + matVy2Vz[ij+3*2+2] * DyVz;
      }
      else // lower than surface, lower order
      {
        M_FD_SHIFT(DzVx, Vx, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);
        M_FD_SHIFT(DzVy, Vy, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);
        M_FD_SHIFT(DzVz, Vz, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);
      }

      // Hooke's equatoin
	    hTxx[iptr] = (c11*xix + c16*xiy + c15*xiz) * DxVx + (c16*xix + c12*xiy + c14*xiz) * DxVy + (c15*xix + c14*xiy + c13*xiz) * DxVz
                 + (c11*etx + c16*ety + c15*etz) * DyVx + (c16*etx + c12*ety + c14*etz) * DyVy + (c15*etx + c14*ety + c13*etz) * DyVz
                 + (c11*ztx + c16*zty + c15*ztz) * DzVx + (c16*ztx + c12*zty + c14*ztz) * DzVy + (c15*ztx + c14*zty + c13*ztz) * DzVz;
      
      hTyy[iptr] = (c12*xix + c26*xiy + c25*xiz) * DxVx + (c26*xix + c22*xiy + c24*xiz) * DxVy + (c25*xix + c24*xiy + c23*xiz) * DxVz
                 + (c12*etx + c26*ety + c25*etz) * DyVx + (c26*etx + c22*ety + c24*etz) * DyVy + (c25*etx + c24*ety + c23*etz) * DyVz
                 + (c12*ztx + c26*zty + c25*ztz) * DzVx + (c26*ztx + c22*zty + c24*ztz) * DzVy + (c25*ztx + c24*zty + c23*ztz) * DzVz;
     
      hTzz[iptr] = (c13*xix + c36*xiy + c35*xiz) * DxVx + (c36*xix + c23*xiy + c34*xiz) * DxVy + (c35*xix + c34*xiy + c33*xiz) * DxVz
                 + (c13*etx + c36*ety + c35*etz) * DyVx + (c36*etx + c23*ety + c34*etz) * DyVy + (c35*etx + c34*ety + c33*etz) * DyVz
                 + (c13*ztx + c36*zty + c35*ztz) * DzVx + (c36*ztx + c23*zty + c34*ztz) * DzVy + (c35*ztx + c34*zty + c33*ztz) * DzVz;
  

      hTyz[iptr] = (c14*xix + c46*xiy + c45*xiz) * DxVx + (c46*xix + c24*xiy + c44*xiz) * DxVy + (c45*xix + c44*xiy + c34*xiz) * DxVz
                 + (c14*etx + c46*ety + c45*etz) * DyVx + (c46*etx + c24*ety + c44*etz) * DyVy + (c45*etx + c44*ety + c34*etz) * DyVz
                 + (c14*ztx + c46*zty + c45*ztz) * DzVx + (c46*ztx + c24*zty + c44*ztz) * DzVy + (c45*ztx + c44*zty + c34*ztz) * DzVz;
  
      hTxz[iptr] = (c15*xix + c56*xiy + c55*xiz) * DxVx + (c56*xix + c25*xiy + c45*xiz) * DxVy + (c55*xix + c45*xiy + c35*xiz) * DxVz
                 + (c15*etx + c56*ety + c55*etz) * DyVx + (c56*etx + c25*ety + c45*etz) * DyVy + (c55*etx + c45*ety + c35*etz) * DyVz
                 + (c15*ztx + c56*zty + c55*ztz) * DzVx + (c56*ztx + c25*zty + c45*ztz) * DzVy + (c55*ztx + c45*zty + c35*ztz) * DzVz;
  
      hTxy[iptr] = (c16*xix + c66*xiy + c56*xiz) * DxVx + (c66*xix + c26*xiy + c46*xiz) * DxVy + (c56*xix + c46*xiy + c36*xiz) * DxVz
                 + (c16*etx + c66*ety + c56*etz) * DyVx + (c66*etx + c26*ety + c46*etz) * DyVy + (c56*etx + c46*ety + c36*etz) * DyVz
                 + (c16*ztx + c66*zty + c56*ztz) * DzVx + (c66*ztx + c26*zty + c46*ztz) * DzVy + (c56*ztx + c46*zty + c36*ztz) * DzVz;
    }
  }

  return;
}

/*******************************************************************************
 * CFS-PML boundary
 ******************************************************************************/

/*
 * cfspml, reference to each pml var inside function
 */

int
sv_curv_col_el_aniso_rhs_cfspml(
    float *  Vx , float *  Vy , float *  Vz ,
    float *  Txx, float *  Tyy, float *  Tzz,
    float *  Txz, float *  Tyz, float *  Txy,
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * xi_x, float * xi_y, float * xi_z,
    float * et_x, float * et_y, float * et_z,
    float * zt_x, float * zt_y, float * zt_z,
    float * c11d, float * c12d, float * c13d,
    float * c14d, float * c15d, float * c16d,
                  float * c22d, float * c23d,
    float * c24d, float * c25d, float * c26d,
                                float * c33d,
    float * c34d, float * c35d, float * c36d,
    float * c44d, float * c45d, float * c46d,
                  float * c55d, float * c56d,
                                float * c66d,
                                float * slw3d,
    int nk2, size_t siz_iy, size_t siz_iz,
    size_t * lfdx_shift, float * lfdx_coef,
    size_t * lfdy_shift, float * lfdy_coef,
    size_t * lfdz_shift, float * lfdz_coef,
    bdrypml_t bdrypml_d, bdryfree_t bdryfree_d,
    const int myid, const int verbose)
{
  // check each side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      // skip to next face if not cfspml
      if (bdrypml_d.is_sides_pml[idim][iside] == 0) continue;

      // get index into local var
      int abs_ni1 = bdrypml_d.ni1[idim][iside];
      int abs_ni2 = bdrypml_d.ni2[idim][iside];
      int abs_nj1 = bdrypml_d.nj1[idim][iside];
      int abs_nj2 = bdrypml_d.nj2[idim][iside];
      int abs_nk1 = bdrypml_d.nk1[idim][iside];
      int abs_nk2 = bdrypml_d.nk2[idim][iside];

      
      int abs_ni = abs_ni2-abs_ni1+1; 
      int abs_nj = abs_nj2-abs_nj1+1; 
      int abs_nk = abs_nk2-abs_nk1+1; 
      {
        dim3 block(8,4,4);
        dim3 grid;
        grid.x = (abs_ni+block.x-1)/block.x;
        grid.y = (abs_nj+block.y-1)/block.y;
        grid.z = (abs_nk+block.z-1)/block.z;

        sv_curv_col_el_aniso_rhs_cfspml_gpu <<<grid, block>>> (
                                idim, iside, Vx , Vy , Vz , 
                                Txx, Tyy, Tzz, Txz, Tyz, Txy, 
                                hVx, hVy, hVz, hTxx, hTyy, hTzz, 
                                hTxz, hTyz, hTxy, xi_x, xi_y, xi_z,
                                et_x, et_y, et_z, zt_x, zt_y, zt_z, 
                                c11d,c12d,c13d,c14d,c15d,c16d,
                                     c22d,c23d,c24d,c25d,c26d,
                                          c33d,c34d,c35d,c36d,
                                               c44d,c45d,c46d,
                                                    c55d,c56d,
                                                    c66d,slw3d,
                                nk2, siz_iy, siz_iz,
                                lfdx_shift,  lfdx_coef,
                                lfdy_shift,  lfdy_coef,
                                lfdz_shift,  lfdz_coef,
                                bdrypml_d, bdryfree_d, 
                                myid, verbose);
        //cudaDeviceSynchronize();
      }
    } // iside
  } // idim

  return 0;
}


__global__ void
sv_curv_col_el_aniso_rhs_cfspml_gpu(int idim, int iside,
                                    float *  Vx , float *  Vy , float *  Vz ,
                                    float *  Txx, float *  Tyy, float *  Tzz,
                                    float *  Txz, float *  Tyz, float *  Txy,
                                    float * hVx , float * hVy , float * hVz ,
                                    float * hTxx, float * hTyy, float * hTzz,
                                    float * hTxz, float * hTyz, float * hTxy,
                                    float * xi_x, float * xi_y, float * xi_z,
                                    float * et_x, float * et_y, float * et_z,
                                    float * zt_x, float * zt_y, float * zt_z,
                                    float * c11d, float * c12d, float * c13d,
                                    float * c14d, float * c15d, float * c16d,
                                                  float * c22d, float * c23d,
                                    float * c24d, float * c25d, float * c26d,
                                                                float * c33d,
                                    float * c34d, float * c35d, float * c36d,
                                    float * c44d, float * c45d, float * c46d,
                                                  float * c55d, float * c56d,
                                                                float * c66d,
                                                                float * slw3d,
                                    int nk2, size_t siz_iy, size_t siz_iz,
                                    size_t * lfdx_shift, float * lfdx_coef,
                                    size_t * lfdy_shift, float * lfdy_coef,
                                    size_t * lfdz_shift, float * lfdz_coef,
                                    bdrypml_t bdrypml_d, bdryfree_t bdryfree_d,
                                    const int myid, const int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  float *matVx2Vz = bdryfree_d.matVx2Vz2;
  float *matVy2Vz = bdryfree_d.matVy2Vz2;
  // local
  size_t iptr, iptr_a;
  float coef_A, coef_B, coef_D, coef_B_minus_1;

  float * Vx_ptr;
  float * Vy_ptr;
  float * Vz_ptr;
  float * Txx_ptr;
  float * Txy_ptr;
  float * Txz_ptr;
  float * Tyy_ptr;
  float * Tzz_ptr;
  float * Tyz_ptr;

  // get index into local var
  int abs_ni1 = bdrypml_d.ni1[idim][iside];
  int abs_ni2 = bdrypml_d.ni2[idim][iside];
  int abs_nj1 = bdrypml_d.nj1[idim][iside];
  int abs_nj2 = bdrypml_d.nj2[idim][iside];
  int abs_nk1 = bdrypml_d.nk1[idim][iside];
  int abs_nk2 = bdrypml_d.nk2[idim][iside];
  
  int abs_ni = abs_ni2-abs_ni1+1; 
  int abs_nj = abs_nj2-abs_nj1+1; 
  int abs_nk = abs_nk2-abs_nk1+1; 

  // val on point
  float DxTxx,DxTyy,DxTzz,DxTxy,DxTxz,DxTyz,DxVx,DxVy,DxVz;
  float DyTxx,DyTyy,DyTzz,DyTxy,DyTxz,DyTyz,DyVx,DyVy,DyVz;
  float DzTxx,DzTyy,DzTzz,DzTxy,DzTxz,DzTyz,DzVx,DzVy,DzVz;
  float slw;
  float c11,c12,c13,c14,c15,c16;
  float     c22,c23,c24,c25,c26;
  float         c33,c34,c35,c36;
  float             c44,c45,c46;
  float                 c55,c56;
  float                     c66;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;
  float hVx_rhs,hVy_rhs,hVz_rhs;
  float hTxx_rhs,hTyy_rhs,hTzz_rhs,hTxz_rhs,hTyz_rhs,hTxy_rhs;
  // for free surface
  float Dx_DzVx,Dy_DzVx,Dx_DzVy,Dy_DzVy,Dx_DzVz,Dy_DzVz;
  // get coef for this face
  float * ptr_coef_A = bdrypml_d.A[idim][iside];
  float * ptr_coef_B = bdrypml_d.B[idim][iside];
  float * ptr_coef_D = bdrypml_d.D[idim][iside];

  bdrypml_auxvar_t *auxvar = &(bdrypml_d.auxvar[idim][iside]);

  // get pml vars
  float * abs_vars_cur = auxvar->cur;
  float * abs_vars_rhs = auxvar->rhs;

  float * pml_Vx   = abs_vars_cur + auxvar->Vx_pos;
  float * pml_Vy   = abs_vars_cur + auxvar->Vy_pos;
  float * pml_Vz   = abs_vars_cur + auxvar->Vz_pos;
  float * pml_Txx  = abs_vars_cur + auxvar->Txx_pos;
  float * pml_Tyy  = abs_vars_cur + auxvar->Tyy_pos;
  float * pml_Tzz  = abs_vars_cur + auxvar->Tzz_pos;
  float * pml_Txz  = abs_vars_cur + auxvar->Txz_pos;
  float * pml_Tyz  = abs_vars_cur + auxvar->Tyz_pos;
  float * pml_Txy  = abs_vars_cur + auxvar->Txy_pos;

  float * pml_hVx  = abs_vars_rhs + auxvar->Vx_pos;
  float * pml_hVy  = abs_vars_rhs + auxvar->Vy_pos;
  float * pml_hVz  = abs_vars_rhs + auxvar->Vz_pos;
  float * pml_hTxx = abs_vars_rhs + auxvar->Txx_pos;
  float * pml_hTyy = abs_vars_rhs + auxvar->Tyy_pos;
  float * pml_hTzz = abs_vars_rhs + auxvar->Tzz_pos;
  float * pml_hTxz = abs_vars_rhs + auxvar->Txz_pos;
  float * pml_hTyz = abs_vars_rhs + auxvar->Tyz_pos;
  float * pml_hTxy = abs_vars_rhs + auxvar->Txy_pos;


  // for each dim
  if (idim == 0 ) // x direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1) * siz_iy + (iz+abs_nk1) * siz_iz;
      // pml coefs
      // int abs_i = ix;
      coef_D = ptr_coef_D[ix];
      coef_A = ptr_coef_A[ix];
      coef_B = ptr_coef_B[ix];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      xix = xi_x[iptr];
      xiy = xi_y[iptr];
      xiz = xi_z[iptr];

      // medium
      slw = slw3d[iptr];
      c11 = c11d[iptr];
      c12 = c12d[iptr];
      c13 = c13d[iptr];
      c14 = c14d[iptr];
      c15 = c15d[iptr];
      c16 = c16d[iptr];
      c22 = c22d[iptr];
      c23 = c23d[iptr];
      c24 = c24d[iptr];
      c25 = c25d[iptr];
      c26 = c26d[iptr];
      c33 = c33d[iptr];
      c34 = c34d[iptr];
      c35 = c35d[iptr];
      c36 = c36d[iptr];
      c44 = c44d[iptr];
      c45 = c45d[iptr];
      c46 = c46d[iptr];
      c55 = c55d[iptr];
      c56 = c56d[iptr];
      c66 = c66d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;

      // xi derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVx , Vx_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxVy , Vy_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxVz , Vz_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTxx, Txx_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTyy, Tyy_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTzz, Tzz_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTxz, Txz_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTyz, Tyz_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxTxy, Txy_ptr, lfdx_shift, lfdx_coef);

      // combine for corr and aux vars
       hVx_rhs = slw * ( xix*DxTxx + xiy*DxTxy + xiz*DxTxz );
       hVy_rhs = slw * ( xix*DxTxy + xiy*DxTyy + xiz*DxTyz );
       hVz_rhs = slw * ( xix*DxTxz + xiy*DxTyz + xiz*DxTzz );
      hTxx_rhs = (c11*xix+c16*xiy+c15*xiz)*DxVx + (c16*xix+c12*xiy+c14*xiz)*DxVy + (c15*xix+c14*xiy+c13*xiz)*DxVz; 
      hTyy_rhs = (c12*xix+c26*xiy+c25*xiz)*DxVx + (c26*xix+c22*xiy+c24*xiz)*DxVy + (c25*xix+c24*xiy+c23*xiz)*DxVz;
      hTzz_rhs = (c13*xix+c36*xiy+c35*xiz)*DxVx + (c36*xix+c23*xiy+c34*xiz)*DxVy + (c35*xix+c34*xiy+c33*xiz)*DxVz;
      hTyz_rhs = (c14*xix+c46*xiy+c45*xiz)*DxVx + (c46*xix+c24*xiy+c44*xiz)*DxVy + (c45*xix+c44*xiy+c34*xiz)*DxVz;
      hTxz_rhs = (c15*xix+c56*xiy+c55*xiz)*DxVx + (c56*xix+c25*xiy+c45*xiz)*DxVy + (c55*xix+c45*xiy+c35*xiz)*DxVz;
      hTxy_rhs = (c16*xix+c66*xiy+c56*xiz)*DxVx + (c66*xix+c26*xiy+c46*xiz)*DxVy + (c56*xix+c46*xiy+c36*xiz)*DxVz;

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];

      // add contributions from free surface condition
      //  not consider timg because conflict with main cfspml,
      //     need to revise in the future if required
      if (bdryfree_d.is_sides_free[CONST_NDIM-1][1]==1 && (iz+abs_nk1)==nk2)
      {
        // zeta derivatives
        size_t ij = ((ix+abs_ni1) + (iy+abs_nj1) * siz_iy)*9;
        Dx_DzVx = matVx2Vz[ij+3*0+0] * DxVx
                + matVx2Vz[ij+3*0+1] * DxVy
                + matVx2Vz[ij+3*0+2] * DxVz;

        Dx_DzVy = matVx2Vz[ij+3*1+0] * DxVx
                + matVx2Vz[ij+3*1+1] * DxVy
                + matVx2Vz[ij+3*1+2] * DxVz;

        Dx_DzVz = matVx2Vz[ij+3*2+0] * DxVx
                + matVx2Vz[ij+3*2+1] * DxVy
                + matVx2Vz[ij+3*2+2] * DxVz;

        // metric
        ztx = zt_x[iptr];
        zty = zt_y[iptr];
        ztz = zt_z[iptr];

        // keep xi derivative terms, including free surface convered
        hTxx_rhs = (c11*ztx+c16*zty+c15*ztz)*Dx_DzVx + (c16*ztx+c12*zty+c14*ztz)*Dx_DzVy + (c15*ztx+c14*zty+c13*ztz)*Dx_DzVz; 
        hTyy_rhs = (c12*ztx+c26*zty+c25*ztz)*Dx_DzVx + (c26*ztx+c22*zty+c24*ztz)*Dx_DzVy + (c25*ztx+c24*zty+c23*ztz)*Dx_DzVz;
        hTzz_rhs = (c13*ztx+c36*zty+c35*ztz)*Dx_DzVx + (c36*ztx+c23*zty+c34*ztz)*Dx_DzVy + (c35*ztx+c34*zty+c33*ztz)*Dx_DzVz;
        hTyz_rhs = (c14*ztx+c46*zty+c45*ztz)*Dx_DzVx + (c46*ztx+c24*zty+c44*ztz)*Dx_DzVy + (c45*ztx+c44*zty+c34*ztz)*Dx_DzVz;
        hTxz_rhs = (c15*ztx+c56*zty+c55*ztz)*Dx_DzVx + (c56*ztx+c25*zty+c45*ztz)*Dx_DzVy + (c55*ztx+c45*zty+c35*ztz)*Dx_DzVz;
        hTxy_rhs = (c16*ztx+c66*zty+c56*ztz)*Dx_DzVx + (c66*ztx+c26*zty+c46*ztz)*Dx_DzVy + (c56*ztx+c46*zty+c36*ztz)*Dx_DzVz;

        // make corr to Hooke's equatoin
        hTxx[iptr] += (coef_B - 1.0) * hTxx_rhs;
        hTyy[iptr] += (coef_B - 1.0) * hTyy_rhs;
        hTzz[iptr] += (coef_B - 1.0) * hTzz_rhs;
        hTxz[iptr] += (coef_B - 1.0) * hTxz_rhs;
        hTyz[iptr] += (coef_B - 1.0) * hTyz_rhs;
        hTxy[iptr] += (coef_B - 1.0) * hTxy_rhs;

        // aux var
        //   a1 = alpha + d / beta, dealt in abs_set_cfspml
        pml_hTxx[iptr_a] += coef_D * hTxx_rhs;
        pml_hTyy[iptr_a] += coef_D * hTyy_rhs;
        pml_hTzz[iptr_a] += coef_D * hTzz_rhs;
        pml_hTxz[iptr_a] += coef_D * hTxz_rhs;
        pml_hTyz[iptr_a] += coef_D * hTyz_rhs;
        pml_hTxy[iptr_a] += coef_D * hTxy_rhs;
      } // if nk2
    }
  }
  else if (idim == 1) // y direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1)*siz_iy + (iz+abs_nk1) * siz_iz;

      // pml coefs
      // int abs_j = iy;
      coef_D = ptr_coef_D[iy];
      coef_A = ptr_coef_A[iy];
      coef_B = ptr_coef_B[iy];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      etx = et_x[iptr];
      ety = et_y[iptr];
      etz = et_z[iptr];

      // medium
      slw = slw3d[iptr];
      c11 = c11d[iptr];
      c12 = c12d[iptr];
      c13 = c13d[iptr];
      c14 = c14d[iptr];
      c15 = c15d[iptr];
      c16 = c16d[iptr];
      c22 = c22d[iptr];
      c23 = c23d[iptr];
      c24 = c24d[iptr];
      c25 = c25d[iptr];
      c26 = c26d[iptr];
      c33 = c33d[iptr];
      c34 = c34d[iptr];
      c35 = c35d[iptr];
      c36 = c36d[iptr];
      c44 = c44d[iptr];
      c45 = c45d[iptr];
      c46 = c46d[iptr];
      c55 = c55d[iptr];
      c56 = c56d[iptr];
      c66 = c66d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;

      // et derivatives
      M_FD_SHIFT_PTR_MACDRP(DyVx , Vx_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyVy , Vy_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyVz , Vz_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTxx, Txx_ptr, lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTyy, Tyy_ptr, lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTzz, Tzz_ptr, lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTxz, Txz_ptr, lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTyz, Tyz_ptr, lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyTxy, Txy_ptr, lfdy_shift, lfdy_coef);

      // combine for corr and aux vars
       hVx_rhs = slw * ( etx*DyTxx + ety*DyTxy + etz*DyTxz );
       hVy_rhs = slw * ( etx*DyTxy + ety*DyTyy + etz*DyTyz );
       hVz_rhs = slw * ( etx*DyTxz + ety*DyTyz + etz*DyTzz );
      hTxx_rhs = (c11*etx+c16*ety+c15*etz)*DyVx + (c16*etx+c12*ety+c14*etz)*DyVy + (c15*etx+c14*ety+c13*etz)*DyVz; 
      hTyy_rhs = (c12*etx+c26*ety+c25*etz)*DyVx + (c26*etx+c22*ety+c24*etz)*DyVy + (c25*etx+c24*ety+c23*etz)*DyVz;
      hTzz_rhs = (c13*etx+c36*ety+c35*etz)*DyVx + (c36*etx+c23*ety+c34*etz)*DyVy + (c35*etx+c34*ety+c33*etz)*DyVz;
      hTyz_rhs = (c14*etx+c46*ety+c45*etz)*DyVx + (c46*etx+c24*ety+c44*etz)*DyVy + (c45*etx+c44*ety+c34*etz)*DyVz;
      hTxz_rhs = (c15*etx+c56*ety+c55*etz)*DyVx + (c56*etx+c25*ety+c45*etz)*DyVy + (c55*etx+c45*ety+c35*etz)*DyVz;
      hTxy_rhs = (c16*etx+c66*ety+c56*etz)*DyVx + (c66*etx+c26*ety+c46*etz)*DyVy + (c56*etx+c46*ety+c36*etz)*DyVz;

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];

      // add contributions from free surface condition
      if (bdryfree_d.is_sides_free[CONST_NDIM-1][1]==1 && (iz+abs_nk1)==nk2)
      {
        // zeta derivatives
        size_t ij = ((ix+abs_ni1) + (iy+abs_nj1) * siz_iy)*9;
        Dy_DzVx = matVy2Vz[ij+3*0+0] * DyVx
                + matVy2Vz[ij+3*0+1] * DyVy
                + matVy2Vz[ij+3*0+2] * DyVz;

        Dy_DzVy = matVy2Vz[ij+3*1+0] * DyVx
                + matVy2Vz[ij+3*1+1] * DyVy
                + matVy2Vz[ij+3*1+2] * DyVz;

        Dy_DzVz = matVy2Vz[ij+3*2+0] * DyVx
                + matVy2Vz[ij+3*2+1] * DyVy
                + matVy2Vz[ij+3*2+2] * DyVz;

        // metric
        ztx = zt_x[iptr];
        zty = zt_y[iptr];
        ztz = zt_z[iptr];

        // keep eta derivative terms, including free surface convered
        hTxx_rhs = (c11*ztx+c16*zty+c15*ztz)*Dy_DzVx + (c16*ztx+c12*zty+c14*ztz)*Dy_DzVy + (c15*ztx+c14*zty+c13*ztz)*Dy_DzVz; 
        hTyy_rhs = (c12*ztx+c26*zty+c25*ztz)*Dy_DzVx + (c26*ztx+c22*zty+c24*ztz)*Dy_DzVy + (c25*ztx+c24*zty+c23*ztz)*Dy_DzVz;
        hTzz_rhs = (c13*ztx+c36*zty+c35*ztz)*Dy_DzVx + (c36*ztx+c23*zty+c34*ztz)*Dy_DzVy + (c35*ztx+c34*zty+c33*ztz)*Dy_DzVz;
        hTyz_rhs = (c14*ztx+c46*zty+c45*ztz)*Dy_DzVx + (c46*ztx+c24*zty+c44*ztz)*Dy_DzVy + (c45*ztx+c44*zty+c34*ztz)*Dy_DzVz;
        hTxz_rhs = (c15*ztx+c56*zty+c55*ztz)*Dy_DzVx + (c56*ztx+c25*zty+c45*ztz)*Dy_DzVy + (c55*ztx+c45*zty+c35*ztz)*Dy_DzVz;
        hTxy_rhs = (c16*ztx+c66*zty+c56*ztz)*Dy_DzVx + (c66*ztx+c26*zty+c46*ztz)*Dy_DzVy + (c56*ztx+c46*zty+c36*ztz)*Dy_DzVz;

        // make corr to Hooke's equatoin
        hTxx[iptr] += (coef_B - 1.0) * hTxx_rhs;
        hTyy[iptr] += (coef_B - 1.0) * hTyy_rhs;
        hTzz[iptr] += (coef_B - 1.0) * hTzz_rhs;
        hTxz[iptr] += (coef_B - 1.0) * hTxz_rhs;
        hTyz[iptr] += (coef_B - 1.0) * hTyz_rhs;
        hTxy[iptr] += (coef_B - 1.0) * hTxy_rhs;

        // aux var
        //   a1 = alpha + d / beta, dealt in abs_set_cfspml
        pml_hTxx[iptr_a] += coef_D * hTxx_rhs;
        pml_hTyy[iptr_a] += coef_D * hTyy_rhs;
        pml_hTzz[iptr_a] += coef_D * hTzz_rhs;
        pml_hTxz[iptr_a] += coef_D * hTxz_rhs;
        pml_hTyz[iptr_a] += coef_D * hTyz_rhs;
        pml_hTxy[iptr_a] += coef_D * hTxy_rhs;
      }
    }
  }
  else // z direction
  {
    if(ix<abs_ni  && iy<abs_nj && iz<abs_nk)
    {
      iptr_a = iz*(abs_nj*abs_ni) + iy*abs_ni + ix;
      iptr   = (ix + abs_ni1) + (iy+abs_nj1) * siz_iy + (iz+abs_nk1) * siz_iz;
      // pml coefs
      // int abs_k = iz;
      coef_D = ptr_coef_D[iz];
      coef_A = ptr_coef_A[iz];
      coef_B = ptr_coef_B[iz];
      coef_B_minus_1 = coef_B - 1.0;

      // metric
      ztx = zt_x[iptr];
      zty = zt_y[iptr];
      ztz = zt_z[iptr];

      // medium
      slw = slw3d[iptr];
      c11 = c11d[iptr];
      c12 = c12d[iptr];
      c13 = c13d[iptr];
      c14 = c14d[iptr];
      c15 = c15d[iptr];
      c16 = c16d[iptr];
      c22 = c22d[iptr];
      c23 = c23d[iptr];
      c24 = c24d[iptr];
      c25 = c25d[iptr];
      c26 = c26d[iptr];
      c33 = c33d[iptr];
      c34 = c34d[iptr];
      c35 = c35d[iptr];
      c36 = c36d[iptr];
      c44 = c44d[iptr];
      c45 = c45d[iptr];
      c46 = c46d[iptr];
      c55 = c55d[iptr];
      c56 = c56d[iptr];
      c66 = c66d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      Txx_ptr = Txx + iptr;
      Tyy_ptr = Tyy + iptr;
      Tzz_ptr = Tzz + iptr;
      Txz_ptr = Txz + iptr;
      Tyz_ptr = Tyz + iptr;
      Txy_ptr = Txy + iptr;

      // zt derivatives
      M_FD_SHIFT_PTR_MACDRP(DzVx , Vx_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzVy , Vy_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzVz , Vz_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTxx, Txx_ptr, lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTyy, Tyy_ptr, lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTzz, Tzz_ptr, lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTxz, Txz_ptr, lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTyz, Tyz_ptr, lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzTxy, Txy_ptr, lfdz_shift, lfdz_coef);

      // combine for corr and aux vars
       hVx_rhs = slw * ( ztx*DzTxx + zty*DzTxy + ztz*DzTxz );
       hVy_rhs = slw * ( ztx*DzTxy + zty*DzTyy + ztz*DzTyz );
       hVz_rhs = slw * ( ztx*DzTxz + zty*DzTyz + ztz*DzTzz );
      hTxx_rhs = (c11*ztx+c16*zty+c15*ztz)*DzVx + (c16*ztx+c12*zty+c14*ztz)*DzVy + (c15*ztx+c14*zty+c13*ztz)*DzVz; 
      hTyy_rhs = (c12*ztx+c26*zty+c25*ztz)*DzVx + (c26*ztx+c22*zty+c24*ztz)*DzVy + (c25*ztx+c24*zty+c23*ztz)*DzVz;
      hTzz_rhs = (c13*ztx+c36*zty+c35*ztz)*DzVx + (c36*ztx+c23*zty+c34*ztz)*DzVy + (c35*ztx+c34*zty+c33*ztz)*DzVz;
      hTyz_rhs = (c14*ztx+c46*zty+c45*ztz)*DzVx + (c46*ztx+c24*zty+c44*ztz)*DzVy + (c45*ztx+c44*zty+c34*ztz)*DzVz;
      hTxz_rhs = (c15*ztx+c56*zty+c55*ztz)*DzVx + (c56*ztx+c25*zty+c45*ztz)*DzVy + (c55*ztx+c45*zty+c35*ztz)*DzVz;
      hTxy_rhs = (c16*ztx+c66*zty+c56*ztz)*DzVx + (c66*ztx+c26*zty+c46*ztz)*DzVy + (c56*ztx+c46*zty+c36*ztz)*DzVz;

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hTxx[iptr] += coef_B_minus_1 * hTxx_rhs - coef_B * pml_Txx[iptr_a];
      hTyy[iptr] += coef_B_minus_1 * hTyy_rhs - coef_B * pml_Tyy[iptr_a];
      hTzz[iptr] += coef_B_minus_1 * hTzz_rhs - coef_B * pml_Tzz[iptr_a];
      hTxz[iptr] += coef_B_minus_1 * hTxz_rhs - coef_B * pml_Txz[iptr_a];
      hTyz[iptr] += coef_B_minus_1 * hTyz_rhs - coef_B * pml_Tyz[iptr_a];
      hTxy[iptr] += coef_B_minus_1 * hTxy_rhs - coef_B * pml_Txy[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hTxx[iptr_a] = coef_D * hTxx_rhs - coef_A * pml_Txx[iptr_a];
      pml_hTyy[iptr_a] = coef_D * hTyy_rhs - coef_A * pml_Tyy[iptr_a];
      pml_hTzz[iptr_a] = coef_D * hTzz_rhs - coef_A * pml_Tzz[iptr_a];
      pml_hTxz[iptr_a] = coef_D * hTxz_rhs - coef_A * pml_Txz[iptr_a];
      pml_hTyz[iptr_a] = coef_D * hTyz_rhs - coef_A * pml_Tyz[iptr_a];
      pml_hTxy[iptr_a] = coef_D * hTxy_rhs - coef_A * pml_Txy[iptr_a];
    } 
  } 

  return;
}

/*******************************************************************************
 * free surface coef
 * converted matrix for velocity gradient
 *  only implement z2 (top) right now
 ******************************************************************************/

__global__ void
sv_curv_col_el_aniso_dvh2dvz_gpu(gdinfo_t        gdinfo_d,
                                     gdcurv_metric_t metric_d,
                                     md_t        md_d,
                                     bdryfree_t  bdryfree_d,
                                     const int verbose)
{
  int ni1 = gdinfo_d.ni1;
  int ni2 = gdinfo_d.ni2;
  int nj1 = gdinfo_d.nj1;
  int nj2 = gdinfo_d.nj2;
  int nk1 = gdinfo_d.nk1;
  int nk2 = gdinfo_d.nk2;
  int nx  = gdinfo_d.nx;
  int ny  = gdinfo_d.ny;
  int nz  = gdinfo_d.nz;
  size_t siz_iy   = gdinfo_d.siz_iy;
  size_t siz_iz   = gdinfo_d.siz_iz;
  size_t siz_icmp = gdinfo_d.siz_icmp;

  // point to each var
  float * xi_x = metric_d.xi_x;
  float * xi_y = metric_d.xi_y;
  float * xi_z = metric_d.xi_z;
  float * et_x = metric_d.eta_x;
  float * et_y = metric_d.eta_y;
  float * et_z = metric_d.eta_z;
  float * zt_x = metric_d.zeta_x;
  float * zt_y = metric_d.zeta_y;
  float * zt_z = metric_d.zeta_z;

  float * c11d = md_d.c11;
  float * c12d = md_d.c12;
  float * c13d = md_d.c13;
  float * c14d = md_d.c14;
  float * c15d = md_d.c15;
  float * c16d = md_d.c16;
  float * c22d = md_d.c22;
  float * c23d = md_d.c23;
  float * c24d = md_d.c24;
  float * c25d = md_d.c25;
  float * c26d = md_d.c26;
  float * c33d = md_d.c33;
  float * c34d = md_d.c34;
  float * c35d = md_d.c35;
  float * c36d = md_d.c36;
  float * c44d = md_d.c44;
  float * c45d = md_d.c45;
  float * c46d = md_d.c46;
  float * c55d = md_d.c55;
  float * c56d = md_d.c56;
  float * c66d = md_d.c66;

  float *matVx2Vz = bdryfree_d.matVx2Vz2;
  float *matVy2Vz = bdryfree_d.matVy2Vz2;

  float A[3][3], B[3][3], C[3][3];
  float AB[3][3], AC[3][3];

  float c11,c12,c13,c14,c15,c16;
  float     c22,c23,c24,c25,c26;
  float         c33,c34,c35,c36;
  float             c44,c45,c46;
  float                 c55,c56;
  float                     c66;
  float xix, xiy ,xiz, etx, ety, etz, ztx, zty, ztz;
 
  int k = nk2;

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  if(ix<(ni2-ni1+1) && iy<(nj2-nj1+1))
  {
    size_t iptr = (ix+ni1) + (iy+nj1) * siz_iy + k * siz_iz;

    xix = xi_x[iptr];
    xiy = xi_y[iptr];
    xiz = xi_z[iptr];
    etx = et_x[iptr];
    ety = et_y[iptr];
    etz = et_z[iptr];
    ztx = zt_x[iptr];
    zty = zt_y[iptr];
    ztz = zt_z[iptr];
    
    c11 = c11d[iptr];
    c12 = c12d[iptr];
    c13 = c13d[iptr];
    c14 = c14d[iptr];
    c15 = c15d[iptr];
    c16 = c16d[iptr];
    c22 = c22d[iptr];
    c23 = c23d[iptr];
    c24 = c24d[iptr];
    c25 = c25d[iptr];
    c26 = c26d[iptr];
    c33 = c33d[iptr];
    c34 = c34d[iptr];
    c35 = c35d[iptr];
    c36 = c36d[iptr];
    c44 = c44d[iptr];
    c45 = c45d[iptr];
    c46 = c46d[iptr];
    c55 = c55d[iptr];
    c56 = c56d[iptr];
    c66 = c66d[iptr];

    // first dim: irow; sec dim: jcol, as Fortran code
    A[0][0] = (c11*ztx+c16*zty+c15*ztz)*ztx + (c16*ztx+c66*zty+c56*ztz)*zty + (c15*ztx+c56*zty+c55*ztz)*ztz;
    A[0][1] = (c16*ztx+c12*zty+c14*ztz)*ztx + (c66*ztx+c26*zty+c46*ztz)*zty + (c56*ztx+c25*zty+c45*ztz)*ztz;
    A[0][2] = (c15*ztx+c14*zty+c13*ztz)*ztx + (c56*ztx+c46*zty+c36*ztz)*zty + (c55*ztx+c45*zty+c35*ztz)*ztz; 
    A[1][0] = (c16*ztx+c66*zty+c56*ztz)*ztx + (c12*ztx+c26*zty+c25*ztz)*zty + (c14*ztx+c46*zty+c45*ztz)*ztz; 
    A[1][1] = (c66*ztx+c26*zty+c46*ztz)*ztx + (c26*ztx+c22*zty+c24*ztz)*zty + (c46*ztx+c24*zty+c44*ztz)*ztz; 
    A[1][2] = (c56*ztx+c46*zty+c36*ztz)*ztx + (c25*ztx+c24*zty+c23*ztz)*zty + (c45*ztx+c44*zty+c34*ztz)*ztz;
    A[2][0] = (c15*ztx+c56*zty+c55*ztz)*ztx + (c14*ztx+c46*zty+c45*ztz)*zty + (c13*ztx+c36*zty+c35*ztz)*ztz;
    A[2][1] = (c56*ztx+c25*zty+c45*ztz)*ztx + (c46*ztx+c24*zty+c44*ztz)*zty + (c36*ztx+c23*zty+c34*ztz)*ztz;
    A[2][2] = (c55*ztx+c45*zty+c35*ztz)*ztx + (c45*ztx+c44*zty+c34*ztz)*zty + (c35*ztx+c34*zty+c33*ztz)*ztz; 
    fdlib_math_invert3x3(A);
                                                     
    B[0][0] = (c11*xix+c16*xiy+c15*xiz)*ztx + (c16*xix+c66*xiy+c56*xiz)*zty + (c15*xix+c56*xiy+c55*xiz)*ztz;
    B[0][1] = (c16*xix+c12*xiy+c14*xiz)*ztx + (c66*xix+c26*xiy+c46*xiz)*zty + (c56*xix+c25*xiy+c45*xiz)*ztz;
    B[0][2] = (c15*xix+c14*xiy+c13*xiz)*ztx + (c56*xix+c46*xiy+c36*xiz)*zty + (c55*xix+c45*xiy+c35*xiz)*ztz; 
    B[1][0] = (c16*xix+c66*xiy+c56*xiz)*ztx + (c12*xix+c26*xiy+c25*xiz)*zty + (c14*xix+c46*xiy+c45*xiz)*ztz; 
    B[1][1] = (c66*xix+c26*xiy+c46*xiz)*ztx + (c26*xix+c22*xiy+c24*xiz)*zty + (c46*xix+c24*xiy+c44*xiz)*ztz; 
    B[1][2] = (c56*xix+c46*xiy+c36*xiz)*ztx + (c25*xix+c24*xiy+c23*xiz)*zty + (c45*xix+c44*xiy+c34*xiz)*ztz;
    B[2][0] = (c15*xix+c56*xiy+c55*xiz)*ztx + (c14*xix+c46*xiy+c45*xiz)*zty + (c13*xix+c36*xiy+c35*xiz)*ztz;
    B[2][1] = (c56*xix+c25*xiy+c45*xiz)*ztx + (c46*xix+c24*xiy+c44*xiz)*zty + (c36*xix+c23*xiy+c34*xiz)*ztz;
    B[2][2] = (c55*xix+c45*xiy+c35*xiz)*ztx + (c45*xix+c44*xiy+c34*xiz)*zty + (c35*xix+c34*xiy+c33*xiz)*ztz; 
     
    C[0][0] = (c11*etx+c16*ety+c15*etz)*ztx + (c16*etx+c66*ety+c56*etz)*zty + (c15*etx+c56*ety+c55*etz)*ztz;
    C[0][1] = (c16*etx+c12*ety+c14*etz)*ztx + (c66*etx+c26*ety+c46*etz)*zty + (c56*etx+c25*ety+c45*etz)*ztz;
    C[0][2] = (c15*etx+c14*ety+c13*etz)*ztx + (c56*etx+c46*ety+c36*etz)*zty + (c55*etx+c45*ety+c35*etz)*ztz; 
    C[1][0] = (c16*etx+c66*ety+c56*etz)*ztx + (c12*etx+c26*ety+c25*etz)*zty + (c14*etx+c46*ety+c45*etz)*ztz; 
    C[1][1] = (c66*etx+c26*ety+c46*etz)*ztx + (c26*etx+c22*ety+c24*etz)*zty + (c46*etx+c24*ety+c44*etz)*ztz; 
    C[1][2] = (c56*etx+c46*ety+c36*etz)*ztx + (c25*etx+c24*ety+c23*etz)*zty + (c45*etx+c44*ety+c34*etz)*ztz;
    C[2][0] = (c15*etx+c56*ety+c55*etz)*ztx + (c14*etx+c46*ety+c45*etz)*zty + (c13*etx+c36*ety+c35*etz)*ztz;
    C[2][1] = (c56*etx+c25*ety+c45*etz)*ztx + (c46*etx+c24*ety+c44*etz)*zty + (c36*etx+c23*ety+c34*etz)*ztz;
    C[2][2] = (c55*etx+c45*ety+c35*etz)*ztx + (c45*etx+c44*ety+c34*etz)*zty + (c35*etx+c34*ety+c33*etz)*ztz; 
    fdlib_math_matmul3x3(A, B, AB);
    fdlib_math_matmul3x3(A, C, AC);

    size_t ij = ((iy+nj1) * siz_iy + (ix+ni1)) * 9;

    // save into mat
    for(int irow = 0; irow < 3; irow++){
      for(int jcol = 0; jcol < 3; jcol++){
        matVx2Vz[ij + irow*3 + jcol] = -1.0f * AB[irow][jcol];
        matVy2Vz[ij + irow*3 + jcol] = -1.0f * AC[irow][jcol];
      }
    }
  }

  return;
}
