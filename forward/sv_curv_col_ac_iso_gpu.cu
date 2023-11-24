/*******************************************************************************
 * solver of isotropic elastic 1st-order eqn using curv grid and collocated scheme
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "sv_curv_col_ac_iso_gpu.h"
#include "cuda_common.h"

/*******************************************************************************
 * perform one stage calculation of rhs
 ******************************************************************************/

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
  const int myid, const int verbose)
{
  // local pointer get each vars
  float *Vx    = w_cur_d + wav_d.Vx_pos ;
  float *Vy    = w_cur_d + wav_d.Vy_pos ;
  float *Vz    = w_cur_d + wav_d.Vz_pos ;
  float *P     = w_cur_d + wav_d.Txx_pos;
  float *hVx   = rhs_d   + wav_d.Vx_pos ; 
  float *hVy   = rhs_d   + wav_d.Vy_pos ; 
  float *hVz   = rhs_d   + wav_d.Vz_pos ; 
  float *hP    = rhs_d   + wav_d.Txx_pos; 

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

  float *kappa3d = md_d.kappa;
  float *slw3d   = md_d.rho;

  // grid size
  int ni1 = gd_d.ni1;
  int ni2 = gd_d.ni2;
  int nj1 = gd_d.nj1;
  int nj2 = gd_d.nj2;
  int nk1 = gd_d.nk1;
  int nk2 = gd_d.nk2;

  int ni  = gd_d.ni;
  int nj  = gd_d.nj;
  int nk  = gd_d.nk;
  int nx  = gd_d.nx;
  int ny  = gd_d.ny;
  int nz  = gd_d.nz;
  size_t siz_iy   = gd_d.siz_iy;
  size_t siz_iz   = gd_d.siz_iz;
  size_t siz_icmp = gd_d.siz_icmp;

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
  
  // free surface at z2 for pressure
  if (bdryfree_d.is_sides_free[2][1] == 1)
  {
    // imaging
    {
      dim3 block(32,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_ac_iso_rhs_timg_z2_gpu  <<<grid, block>>> (
                          P,ni,nj,ni1,nj1,nk2,nz,
                          siz_iy,siz_iz,
                          myid, verbose);
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  {
    dim3 block(32,4,2);
    dim3 grid;
    grid.x = (ni+block.x-1)/block.x;
    grid.y = (nj+block.y-1)/block.y;
    grid.z = (nk+block.z-1)/block.z;

    // inner points
    sv_curv_col_ac_iso_rhs_inner_gpu  <<<grid, block>>> (
                       Vx, Vy, Vz, P, 
                       hVx, hVy, hVz, hP, 
                       xi_x, xi_y, xi_z,
                       et_x, et_y, et_z,
                       zt_x, zt_y, zt_z,
                       kappa3d, slw3d,
                       ni1, ni, nj1, nj, nk1, nk,
                       siz_iy, siz_iz,
                       lfdx_shift_d, lfdx_coef_d,
                       lfdy_shift_d, lfdy_coef_d,
                       lfdz_shift_d, lfdz_coef_d,
                       myid, verbose);
    CUDACHECK(cudaDeviceSynchronize());
  }

  if (bdryfree_d.is_sides_free[2][1] == 1)
  {
    // velocity: vlow
    {
      dim3 block(32,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_ac_iso_rhs_vlow_z2_gpu  <<<grid, block>>> (
                         Vx,Vy,Vz,hP,
                         xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                         kappa3d, slw3d,
                         ni1,ni,nj1,nj,nk1,nk2,siz_iy,siz_iz,
                         fdx_len, lfdx_shift_d, lfdx_coef_d,
                         fdy_len, lfdy_shift_d, lfdy_coef_d,
                         num_of_fdz_op,fdz_max_len,lfdz_len_d,
                         lfdz_coef_all_d,lfdz_shift_all_d,
                         myid, verbose);
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  // cfs-pml, loop face inside
  if (bdrypml_d.is_enable_pml == 1)
  {
    sv_curv_col_ac_iso_rhs_cfspml(Vx,Vy,Vz,P,
                                  hVx,hVy,hVz,hP,
                                  xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                  kappa3d, slw3d,
                                  siz_iy, siz_iz,
                                  lfdx_shift_d, lfdx_coef_d,
                                  lfdy_shift_d, lfdy_coef_d,
                                  lfdz_shift_d, lfdz_coef_d,
                                  bdrypml_d,
                                  myid, verbose);
    
  }

  // add source term
  if (src_d.total_number > 0)
  {
    {
      dim3 block(256);
      dim3 grid;
      grid.x = (src_d.total_number+block.x-1) / block.x;
      sv_curv_col_ac_iso_rhs_src_gpu  <<< grid,block >>> (
                                 hVx,hVy,hVz,hP,
                                 jac3d, slw3d, 
                                 src_d,
                                 myid, verbose);
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return 0;
}

/*******************************************************************************
 * calculate all points without boundaries treatment
 ******************************************************************************/

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
    int myid, int verbose)
{
  // local var
  float DxP,DxVx,DxVy,DxVz;
  float DyP,DyVx,DyVy,DyVz;
  float DzP,DzVx,DzVy,DzVz;
  float kappa,slw;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  float *Vx_ptr;
  float *Vy_ptr;
  float *Vz_ptr;
  float *P_ptr;
  size_t iptr;

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  // caclu all points
  if(ix<ni && iy<nj && iz<nk)
  {
    iptr = (ix+ni1) + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;
    Vx_ptr = Vx + iptr;
    Vy_ptr = Vy + iptr;
    Vz_ptr = Vz + iptr;
    P_ptr  = P  + iptr;

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

    // P derivatives
    M_FD_SHIFT_PTR_MACDRP(DxP, P_ptr, lfdx_shift, lfdx_coef);
    M_FD_SHIFT_PTR_MACDRP(DyP, P_ptr, lfdy_shift, lfdy_coef);
    M_FD_SHIFT_PTR_MACDRP(DzP, P_ptr, lfdz_shift, lfdz_coef);

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
    kappa = kappa3d[iptr];
    slw = slw3d[iptr];

    // moment equation
    hVx[iptr] = - slw*(xix*DxP + etx*DyP + ztx*DzP);
    hVy[iptr] = - slw*(xiy*DxP + ety*DyP + zty*DzP);
    hVz[iptr] = - slw*(xiz*DxP + etz*DyP + ztz*DzP);

    // Hooke's equatoin
    hP[iptr] = -kappa *  ( xix*DxVx + etx*DyVx + ztx*DzVx
                          +xiy*DxVy + ety*DyVy + zty*DzVy
                          +xiz*DxVz + etz*DyVz + ztz*DzVz);

  }

  return;
}

/*******************************************************************************
 * free surface boundary
 ******************************************************************************/

/*
 * implement traction image boundary 
 */

__global__ void
sv_curv_col_ac_iso_rhs_timg_z2_gpu(
                   float *P, int ni, int nj, int ni1,  
                   int nj1, int nk2, int nz, 
                   size_t siz_iy, size_t siz_iz,
                   int myid, int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr;
  size_t iptr_gho;
  size_t iptr_phy;
  int k_phy;

  if(ix<ni && iy<nj)
  {
    iptr = (ix+ni1) + (iy+nj1)*siz_iy + nk2*siz_iz;
    P[iptr] = 0.0;

    // mirror point
    for (int k=nk2+1; k<nz; k++)
    {
      k_phy = 2*nk2 - k;
      iptr_gho = (ix+ni1) + (iy+nj1) * siz_iy + k     * siz_iz;
      iptr_phy = (ix+ni1) + (iy+nj1) * siz_iy + k_phy * siz_iz;

      P[iptr_gho] = -P[iptr_phy];
    }
  }

  return;
}

/*
 * implement vlow boundary
 */

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
    int myid, int verbose)
{

  // local var
  int k;
  int n_fd; // loop var for fd
  int lfdz_len;
  size_t iptr;

  float DxVx,DxVy,DxVz;
  float DyVx,DyVy,DyVz;
  float DzVx,DzVy,DzVz;
  float kappa;
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
      iptr = (ix+ni1) + (iy+nj1) * siz_iy + k * siz_iz;

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
      kappa = kappa3d[iptr];

      // Vx derivatives
      M_FD_SHIFT(DxVx, Vx, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVx, Vx, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      // Vy derivatives
      M_FD_SHIFT(DxVy, Vy, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVy, Vy, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      // Vz derivatives
      M_FD_SHIFT(DxVz, Vz, iptr, fdx_len, lfdx_shift, lfdx_coef, n_fd);
      M_FD_SHIFT(DyVz, Vz, iptr, fdy_len, lfdy_shift, lfdy_coef, n_fd);

      if (k==nk2) // at surface, zero
      {
        hP[iptr] =  0.0;
      }
      else // lower than surface, lower order
      {
        M_FD_SHIFT(DzVx, Vx, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);
        M_FD_SHIFT(DzVy, Vy, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);
        M_FD_SHIFT(DzVz, Vz, iptr, lfdz_len, lfdz_shift, lfdz_coef, n_fd);

        hP[iptr] = -kappa *  ( xix*DxVx  +etx*DyVx + ztx*DzVx
                              +xiy*DxVy + ety*DyVy + zty*DzVy
                              +xiz*DxVz + etz*DyVz + ztz*DzVz);
      }
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
    int myid, int verbose)
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
        dim3 block(32,4,2);
        dim3 grid;
        grid.x = (abs_ni+block.x-1)/block.x;
        grid.y = (abs_nj+block.y-1)/block.y;
        grid.z = (abs_nk+block.z-1)/block.z;

        sv_curv_col_ac_iso_rhs_cfspml_gpu <<<grid, block>>> (
                                idim, iside,
                                Vx , Vy , Vz , P,
                                hVx , hVy , hVz, hP,
                                xi_x, xi_y, xi_z, et_x, et_y, et_z,
                                zt_x, zt_y, zt_z, kappa3d, slw3d,
                                siz_iy, siz_iz,
                                lfdx_shift,  lfdx_coef,
                                lfdy_shift,  lfdy_coef,
                                lfdz_shift,  lfdz_coef,
                                bdrypml_d,
                                myid, verbose);
        cudaDeviceSynchronize();
      }
    } // iside
  } // idim

  return 0;
}

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
                                  int myid, int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

  // local
  size_t iptr, iptr_a;
  float coef_A, coef_B, coef_D, coef_B_minus_1;

  float *Vx_ptr;
  float *Vy_ptr;
  float *Vz_ptr;
  float *P_ptr;

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
  float DxP,DxVx,DxVy,DxVz;
  float DyP,DyVx,DyVy,DyVz;
  float DzP,DzVx,DzVy,DzVz;
  float kappa,slw;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;
  float hVx_rhs,hVy_rhs,hVz_rhs;
  float hP_rhs;

  // get coef for this face
  float *ptr_coef_A = bdrypml_d.A[idim][iside];
  float *ptr_coef_B = bdrypml_d.B[idim][iside];
  float *ptr_coef_D = bdrypml_d.D[idim][iside];

  bdrypml_auxvar_t *auxvar = &(bdrypml_d.auxvar[idim][iside]);

  // get pml vars
  float *abs_vars_cur = auxvar->cur;
  float *abs_vars_rhs = auxvar->rhs;

  float *pml_Vx = abs_vars_cur + auxvar->Vx_pos;
  float *pml_Vy = abs_vars_cur + auxvar->Vy_pos;
  float *pml_Vz = abs_vars_cur + auxvar->Vz_pos;
  float *pml_P  = abs_vars_cur + auxvar->Txx_pos;

  float *pml_hVx = abs_vars_rhs + auxvar->Vx_pos;
  float *pml_hVy = abs_vars_rhs + auxvar->Vy_pos;
  float *pml_hVz = abs_vars_rhs + auxvar->Vz_pos;
  float *pml_hP  = abs_vars_rhs + auxvar->Txx_pos;

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
      kappa  =  kappa3d[iptr];
      slw = slw3d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      P_ptr  = P + iptr;

      // xi derivatives
      M_FD_SHIFT_PTR_MACDRP(DxVx, Vx_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxVy, Vy_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxVz, Vz_ptr, lfdx_shift, lfdx_coef);
      M_FD_SHIFT_PTR_MACDRP(DxP,  P_ptr,  lfdx_shift, lfdx_coef);

      // combine for corr and aux vars
      hVx_rhs = -slw * (xix*DxP                      );
      hVy_rhs = -slw * (            xiy*DxP          );
      hVz_rhs = -slw * (                      xiz*DxP);
      hP_rhs = -kappa* (xix*DxVx + xiy*DxVy + xiz*DxVz);

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hP[iptr] += coef_B_minus_1 * hP_rhs - coef_B * pml_P[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a] = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a] = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a] = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hP[iptr_a] = coef_D * hP_rhs - coef_A * pml_P[iptr_a];
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
      kappa = kappa3d[iptr];
      slw = slw3d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      P_ptr = P + iptr;
      
      // et derivatives
      M_FD_SHIFT_PTR_MACDRP(DyVx, Vx_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyVy, Vy_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyVz, Vz_ptr,  lfdy_shift, lfdy_coef);
      M_FD_SHIFT_PTR_MACDRP(DyP,  P_ptr,   lfdy_shift, lfdy_coef);

      // combine for corr and aux vars
       hVx_rhs = -slw * (etx*DyP                        );
       hVy_rhs = -slw * (            ety*DyP            );
       hVz_rhs = -slw * (                        etz*DyP);
       hP_rhs = -kappa* (etx*DyVx + ety*DyVy + etz*DyVz);

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hP[iptr] += coef_B_minus_1 * hP_rhs - coef_B * pml_P[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hP[iptr_a] = coef_D * hP_rhs - coef_A * pml_P[iptr_a];
    } // k
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
      kappa = kappa3d[iptr];
      slw = slw3d[iptr];

      Vx_ptr = Vx + iptr;
      Vy_ptr = Vy + iptr;
      Vz_ptr = Vz + iptr;
      P_ptr = P + iptr;

      // zt derivatives
      M_FD_SHIFT_PTR_MACDRP(DzVx, Vx_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzVy, Vy_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzVz, Vz_ptr,  lfdz_shift, lfdz_coef);
      M_FD_SHIFT_PTR_MACDRP(DzP,  P_ptr,   lfdz_shift, lfdz_coef);

      // combine for corr and aux vars
       hVx_rhs = -slw * (ztx*DzP                        );
       hVy_rhs = -slw * (            zty*DzP            );
       hVz_rhs = -slw * (                        ztz*DzP);
       hP_rhs = -kappa* (ztx*DzVx + zty*DzVy + ztz*DzVz);

      // 1: make corr to moment equation
      hVx[iptr] += coef_B_minus_1 * hVx_rhs - coef_B * pml_Vx[iptr_a];
      hVy[iptr] += coef_B_minus_1 * hVy_rhs - coef_B * pml_Vy[iptr_a];
      hVz[iptr] += coef_B_minus_1 * hVz_rhs - coef_B * pml_Vz[iptr_a];

      // make corr to Hooke's equatoin
      hP[iptr] += coef_B_minus_1 * hP_rhs - coef_B * pml_P[iptr_a];
      
      // 2: aux var
      //   a1 = alpha + d / beta, dealt in abs_set_cfspml
      pml_hVx[iptr_a]  = coef_D * hVx_rhs  - coef_A * pml_Vx[iptr_a];
      pml_hVy[iptr_a]  = coef_D * hVy_rhs  - coef_A * pml_Vy[iptr_a];
      pml_hVz[iptr_a]  = coef_D * hVz_rhs  - coef_A * pml_Vz[iptr_a];
      pml_hP[iptr_a] = coef_D * hP_rhs - coef_A * pml_P[iptr_a];
    } // k
  } // if which dim

  return;
}

/*******************************************************************************
 * add source terms
 ******************************************************************************/

__global__ void
sv_curv_col_ac_iso_rhs_src_gpu(
    float *hVx , float *hVy , float *hVz ,
    float *hP, 
    float *jac3d, float *slw3d,
    src_t src, 
    int myid, int verbose)
{
  // for easy coding and efficiency
  int max_ext = src.max_ext;

  // get fi / mij
  float fx, fy, fz;
  float Mii;

  int it     = src.it;
  int istage = src.istage;
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;

  // add src; is is a commont iterater var
  if(ix<src.total_number)
  {
    int it_start = src.it_begin[ix];
    int it_end   = src.it_end  [ix];

    if (it >= it_start && it <= it_end)
    {
      int   *ptr_ext_indx = src.ext_indx + ix * max_ext;
      float *ptr_ext_coef = src.ext_coef + ix * max_ext;
      int it_to_it_start = it - it_start;
      size_t iptr_cur_stage =   ix * src.max_nt * src.max_stage // skip other src
                           + it_to_it_start * src.max_stage // skip other time step
                           + istage;
      if (src.force_actived == 1) {
        fx = src.Fx[iptr_cur_stage];
        fy = src.Fy[iptr_cur_stage];
        fz = src.Fz[iptr_cur_stage];
      }
      if (src.moment_actived == 1) {
        Mii = src.Mxx[iptr_cur_stage];
      }
      
      // for extend points
      for (int i_ext=0; i_ext < src.ext_num[ix]; i_ext++)
      {
        int   iptr = ptr_ext_indx[i_ext];
        float coef = ptr_ext_coef[i_ext];
        if (src.force_actived == 1) {
          float V = coef * slw3d[iptr] / jac3d[iptr];
          atomicAdd(&hVx[iptr], fx * V);
          atomicAdd(&hVy[iptr], fy * V);
          atomicAdd(&hVz[iptr], fz * V);
        }

        if (src.moment_actived == 1) {
          float rjac = coef / jac3d[iptr];
          atomicAdd(&hP[iptr], -Mii * rjac);
        }
      } // i_ext
    } // it
  } 

  return;
}

