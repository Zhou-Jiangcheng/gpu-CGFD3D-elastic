/*******************************************************************************
 * solver of isotropic visco-elastic 1st-order eqn using curv grid and collocated scheme
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "sv_curv_col_el_gpu.h"
#include "sv_curv_col_el_iso_gpu.h"
#include "sv_curv_col_vis_iso_gpu.h"
#include "cuda_common.h"

/*******************************************************************************
 * perform one stage calculation of rhs
 ******************************************************************************/

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
  const int myid, const int verbose)
{
  int nmaxwell = md_d.nmaxwell;

  // local pointer get each vars
  float *Vx   = w_cur + wav_d.Vx_pos ;
  float *Vy   = w_cur + wav_d.Vy_pos ;
  float *Vz   = w_cur + wav_d.Vz_pos ;
  float *Txx  = w_cur + wav_d.Txx_pos;
  float *Tyy  = w_cur + wav_d.Tyy_pos;
  float *Tzz  = w_cur + wav_d.Tzz_pos;
  float *Txz  = w_cur + wav_d.Txz_pos;
  float *Tyz  = w_cur + wav_d.Tyz_pos;
  float *Txy  = w_cur + wav_d.Txy_pos;
  float *hVx  = rhs   + wav_d.Vx_pos ; 
  float *hVy  = rhs   + wav_d.Vy_pos ; 
  float *hVz  = rhs   + wav_d.Vz_pos ; 
  float *hTxx = rhs   + wav_d.Txx_pos; 
  float *hTyy = rhs   + wav_d.Tyy_pos; 
  float *hTzz = rhs   + wav_d.Tzz_pos; 
  float *hTxz = rhs   + wav_d.Txz_pos; 
  float *hTyz = rhs   + wav_d.Tyz_pos; 
  float *hTxy = rhs   + wav_d.Txy_pos; 

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

  float *lam3d = md_d.lambda;
  float *mu3d  = md_d.mu;
  float *slw3d = md_d.rho;

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
    dim3 block(32,4,2);
    dim3 grid;
    grid.x = (ni+block.x-1)/block.x;
    grid.y = (nj+block.y-1)/block.y;
    grid.z = (nk+block.z-1)/block.z;
    sv_curv_col_el_iso_rhs_inner_gpu <<<grid, block>>> (
                        Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                        hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        lam3d, mu3d, slw3d,
                        ni1,ni,nj1,nj,nk1,nk,siz_iy,siz_iz,
                        lfdx_shift_d, lfdx_coef_d,
                        lfdy_shift_d, lfdy_coef_d,
                        lfdz_shift_d, lfdz_coef_d,
                        myid, verbose);
    CUDACHECK( cudaDeviceSynchronize() );
  }

  // free, abs, source in turn
  // free surface at z2
  if (bdryfree_d.is_sides_free[2][1] == 1)
  {
    // tractiong
    {
      dim3 block(32,8);
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
      dim3 block(32,8);
      dim3 grid;
      grid.x = (ni+block.x-1)/block.x;
      grid.y = (nj+block.y-1)/block.y;
      sv_curv_col_el_iso_rhs_vlow_z2_gpu  <<<grid, block>>> (
                        Vx,Vy,Vz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                        xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                        lam3d, mu3d, slw3d,
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
    sv_curv_col_el_iso_rhs_cfspml(Vx,Vy,Vz,Txx,Tyy,Tzz,Txz,Tyz,Txy,
                                  hVx,hVy,hVz,hTxx,hTyy,hTzz,hTxz,hTyz,hTxy,
                                  xi_x, xi_y, xi_z, et_x, et_y, et_z, zt_x, zt_y, zt_z,
                                  lam3d, mu3d, slw3d,
                                  nk2, siz_iy,siz_iz,
                                  lfdx_shift_d, lfdx_coef_d,
                                  lfdy_shift_d, lfdy_coef_d,
                                  lfdz_shift_d, lfdz_coef_d,
                                  bdrypml_d, bdryfree_d,
                                  myid, verbose);
  }

  {
    dim3 block(32,4,2);
    dim3 grid;
    grid.x = (ni+block.x-1)/block.x;
    grid.y = (nj+block.y-1)/block.y;
    grid.z = (nk+block.z-1)/block.z;
    sv_curv_col_vis_iso_atten_gpu <<<grid, block>>> (
                w_cur, rhs, wav_d, md_d,
                ni1, ni, nj1, nj, nk1, nk,
                siz_iy, siz_iz,
                myid, verbose);
    CUDACHECK( cudaDeviceSynchronize() );
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

  return 0;
}


/*******************************************************************************
 * add the attenuation term
******************************************************************************/
__global__ void
sv_curv_col_vis_iso_atten_gpu(
    float *w_cur,
    float *rhs, 
    wav_t  wav_d,
    md_t md_d,
    int ni1, int ni, int nj1, int nj, int nk1, int nk,
    size_t siz_iy, size_t siz_iz,
    const int myid, const int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr;

  int nmaxwell = md_d.nmaxwell;

  float *hTxx = rhs + wav_d.Txx_pos; 
  float *hTyy = rhs + wav_d.Tyy_pos; 
  float *hTzz = rhs + wav_d.Tzz_pos; 
  float *hTxz = rhs + wav_d.Txz_pos; 
  float *hTyz = rhs + wav_d.Tyz_pos; 
  float *hTxy = rhs + wav_d.Txy_pos; 
  // number maxwell usually is 3.
  // here we give 10, so the max input 
  // number of maxwell need <= 10
  float *Jxx [10];  
  float *Jyy [10];
  float *Jzz [10];
  float *Jxy [10];
  float *Jxz [10];
  float *Jyz [10];
  float *hJxx[10];
  float *hJyy[10];
  float *hJzz[10];
  float *hJxy[10];
  float *hJxz[10];
  float *hJyz[10];

  float *Ylam[10];
  float *Ymu [10];
  float *wl = md_d.wl;

  for(int i=0; i < nmaxwell; i++)
  {
    Jxx[i]  = w_cur + wav_d.Jxx_pos[i];
    Jyy[i]  = w_cur + wav_d.Jyy_pos[i];
    Jzz[i]  = w_cur + wav_d.Jzz_pos[i];
    Jxy[i]  = w_cur + wav_d.Jxy_pos[i];
    Jyz[i]  = w_cur + wav_d.Jyz_pos[i];
    Jxz[i]  = w_cur + wav_d.Jxz_pos[i];
    hJxx[i] = rhs   + wav_d.Jxx_pos[i];
    hJyy[i] = rhs   + wav_d.Jyy_pos[i];
    hJzz[i] = rhs   + wav_d.Jzz_pos[i];
    hJxy[i] = rhs   + wav_d.Jxy_pos[i];
    hJyz[i] = rhs   + wav_d.Jyz_pos[i];
    hJxz[i] = rhs   + wav_d.Jxz_pos[i];
    Ylam[i] = md_d.Ylam[i];
    Ymu[i]  = md_d.Ymu[i];
  }

  float *lam3d = md_d.lambda;
  float *mu3d  = md_d.mu;
  float *slw3d = md_d.rho;

  float lam,mu;
  float mem_Txx,mem_Tyy,mem_Tzz,mem_Txy,mem_Txz,mem_Tyz;
  float sum_Jxyz,sum_Jxx,sum_Jyy,sum_Jzz,sum_Jxy,sum_Jxz,sum_Jyz;
  float EVxx,EVyy,EVzz,EVxy,EVxz,EVyz;
  float sum_hxyz;

  // caclu all points
  if(ix<ni && iy<nj && iz<nk)
  {
    iptr = (ix+ni1) + (iy+nj1) * siz_iy + (iz+nk1) * siz_iz;

    // medium
    lam = lam3d[iptr];
    mu  =  mu3d[iptr];

    sum_hxyz = (hTxx[iptr]+hTyy[iptr]+hTzz[iptr])/(3*lam+2*mu);

    EVxx = ((2.0*hTxx[iptr]-hTyy[iptr]-hTzz[iptr])/(2*mu) + sum_hxyz)/3;

    EVyy = ((2.0*hTyy[iptr]-hTxx[iptr]-hTzz[iptr])/(2*mu) + sum_hxyz)/3;

    EVzz = ((2.0*hTzz[iptr]-hTxx[iptr]-hTyy[iptr])/(2*mu) + sum_hxyz)/3;

    EVxy = hTxy[iptr]/mu*0.5;
    EVxz = hTxz[iptr]/mu*0.5;
    EVyz = hTyz[iptr]/mu*0.5;
    
    for(int i=0; i<nmaxwell; i++)
    {
      *(hJxx[i]+iptr) = wl[i] * (EVxx - *(Jxx[i]+iptr));
      *(hJyy[i]+iptr) = wl[i] * (EVyy - *(Jyy[i]+iptr));
      *(hJzz[i]+iptr) = wl[i] * (EVzz - *(Jzz[i]+iptr));
      *(hJxy[i]+iptr) = wl[i] * (EVxy - *(Jxy[i]+iptr));
      *(hJxz[i]+iptr) = wl[i] * (EVxz - *(Jxz[i]+iptr));
      *(hJyz[i]+iptr) = wl[i] * (EVyz - *(Jyz[i]+iptr));
    }

    // sum of memory variable for attenuation
    sum_Jxyz = 0.0;
    sum_Jxx = 0.0;
    sum_Jyy = 0.0;
    sum_Jzz = 0.0;
    sum_Jxy = 0.0;
    sum_Jxz = 0.0;
    sum_Jyz = 0.0;
    
    for(int i=0; i<nmaxwell; i++)
    {
      sum_Jxyz += *(Ylam[i]+iptr)  * (*(Jxx[i]+iptr) + *(Jyy[i]+iptr) + *(Jzz[i]+iptr));
      sum_Jxx  += *(Ymu [i]+iptr)  *  *(Jxx[i]+iptr);
      sum_Jyy  += *(Ymu [i]+iptr)  *  *(Jyy[i]+iptr);
      sum_Jzz  += *(Ymu [i]+iptr)  *  *(Jzz[i]+iptr);
      sum_Jxy  += *(Ymu [i]+iptr)  *  *(Jxy[i]+iptr);
      sum_Jxz  += *(Ymu [i]+iptr)  *  *(Jxz[i]+iptr);
      sum_Jyz  += *(Ymu [i]+iptr)  *  *(Jyz[i]+iptr);
    }
    
    mem_Txx = lam*sum_Jxyz + 2.0*mu*sum_Jxx;
    mem_Tyy = lam*sum_Jxyz + 2.0*mu*sum_Jyy;
    mem_Tzz = lam*sum_Jxyz + 2.0*mu*sum_Jzz;
    mem_Txy = 2.0*mu*sum_Jxy;
    mem_Txz = 2.0*mu*sum_Jxz;
    mem_Tyz = 2.0*mu*sum_Jyz;

    hTxx[iptr] -= mem_Txx;
    hTyy[iptr] -= mem_Tyy;
    hTzz[iptr] -= mem_Tzz;
    hTxy[iptr] -= mem_Txy;
    hTxz[iptr] -= mem_Txz;
    hTyz[iptr] -= mem_Tyz;
  }

  return;
}

__global__ void
sv_curv_col_vis_iso_free_gpu(float *w_end,
                             wav_t  wav_d,
                             gd_t   gd_d,
                             gd_metric_t  metric_d,
                             md_t md_d,
                             bdryfree_t  bdryfree_d,
                             const int myid, 
                             const int verbose)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  float *Txx = w_end + wav_d.Txx_pos;
  float *Tyy = w_end + wav_d.Tyy_pos;
  float *Txy = w_end + wav_d.Txy_pos;
  float *Tzz = w_end + wav_d.Tzz_pos;
  float *Txz = w_end + wav_d.Txz_pos;
  float *Tyz = w_end + wav_d.Tyz_pos;

  float *zt_x  = metric_d.zeta_x;
  float *zt_y  = metric_d.zeta_y;
  float *zt_z  = metric_d.zeta_z;

  float *lam3d = md_d.lambda;
  float * mu3d = md_d.mu;

  float *matD = bdryfree_d.matD;

  int ni1 = gd_d.ni1;
  int nj1 = gd_d.nj1;
  int ni  = gd_d.ni;
  int nj  = gd_d.nj;
  int nk2 = gd_d.nk2;

  size_t siz_iy = gd_d.siz_iy;
  size_t siz_iz = gd_d.siz_iz;
  size_t iptr;

  float D[3][3], DT[3][3], Tl[3][3], DTl[3][3], Tg[3][3];
  float d11,d12,d13,d21,d22,d23,d31,d32,d33;
  float lam,mu,lam2mu;
  float tzz;

  if(ix<ni && iy<nj)
  {
    iptr = (ix + ni1) + (iy+nj1) * siz_iy + nk2 * siz_iz;

    size_t ij = ((ix+ni1) + (iy+nj1) * siz_iy)*9;

    lam = lam3d[iptr];
    mu  =  mu3d[iptr];
    lam2mu = lam + 2.0 * mu;

    D[0][0] = matD[ij+3*0+0];
    D[0][1] = matD[ij+3*0+1];
    D[0][2] = matD[ij+3*0+2];
    D[1][0] = matD[ij+3*1+0];
    D[1][1] = matD[ij+3*1+1];
    D[1][2] = matD[ij+3*1+2];
    D[2][0] = matD[ij+3*2+0];
    D[2][1] = matD[ij+3*2+1];
    D[2][2] = matD[ij+3*2+2];

    DT[0][0] = D[0][0];
    DT[0][1] = D[1][0];
    DT[0][2] = D[2][0];
    DT[1][0] = D[0][1];
    DT[1][1] = D[1][1];
    DT[1][2] = D[2][1];
    DT[2][0] = D[0][2];
    DT[2][1] = D[1][2];
    DT[2][2] = D[2][2];

    d11 = D[0][0];
    d12 = D[0][1];
    d13 = D[0][2];
    d21 = D[1][0];
    d22 = D[1][1];
    d23 = D[1][2];
    d31 = D[2][0];
    d32 = D[2][1];
    d33 = D[2][2];

    Tl[0][0] =    d11*d11*Txx[iptr] + d12*d12*Tyy[iptr] + d13*d13*Tzz[iptr]
             + 2*(d11*d12*Txy[iptr] + d11*d13*Txz[iptr] + d12*d13*Tyz[iptr]);

    Tl[0][1] =  d11*d21*Txx[iptr] + d12*d22*Tyy[iptr] + d13*d23*Tzz[iptr]
             + (d11*d22+d12*d21)*Txy[iptr] + (d11*d23+d21*d13)*Txz[iptr]
             + (d12*d23+d22*d13)*Tyz[iptr];
    
    Tl[1][1] =    d21*d21*Txx[iptr] + d22*d22*Tyy[iptr] + d23*d23*Tzz[iptr]
             + 2*(d21*d22*Txy[iptr] + d21*d23*Txz[iptr] + d22*d23*Tyz[iptr]);

    Tl[1][0] = Tl[0][1];
    Tl[0][2] = 0.0;
    Tl[1][2] = 0.0;
    Tl[2][0] = 0.0;
    Tl[2][1] = 0.0;
    Tl[2][2] = 0.0;

    fdlib_math_matmul3x3(DT, Tl, DTl);
    fdlib_math_matmul3x3(DTl, D, Tg);

    Txx[iptr] = Tg[0][0];
    Tyy[iptr] = Tg[1][1];
    Tzz[iptr] = Tg[2][2];
    Txy[iptr] = Tg[0][1];
    Txz[iptr] = Tg[0][2];
    Tyz[iptr] = Tg[1][2];
  }

  return;
}

/*******************************************************************************
 * free surface coef
 * converted matrix for velocity gradient
 *  only implement z2 (top) right now
 ******************************************************************************/
int
sv_curv_col_vis_iso_dvh2dvz(gd_t            *gd,
                            gd_metric_t     *metric,
                            md_t            *md,
                            bdryfree_t      *bdryfree,
                            int fd_len,
                            int *fd_indx,
                            float *fd_coef,
                            const int verbose)
{
  int ni1 = gd->ni1;
  int ni2 = gd->ni2;
  int nj1 = gd->nj1;
  int nj2 = gd->nj2;
  int nk1 = gd->nk1;
  int nk2 = gd->nk2;
  int nx  = gd->nx;
  int ny  = gd->ny;
  int nz  = gd->nz;
  size_t siz_iy = gd->siz_iy;
  size_t siz_iz = gd->siz_iz;

  // point to each var
  float *x3d = gd->x3d;
  float *y3d = gd->y3d;
  float *z3d = gd->z3d;

  float *xi_x = metric->xi_x;
  float *xi_y = metric->xi_y;
  float *xi_z = metric->xi_z;
  float *et_x = metric->eta_x;
  float *et_y = metric->eta_y;
  float *et_z = metric->eta_z;
  float *zt_x = metric->zeta_x;
  float *zt_y = metric->zeta_y;
  float *zt_z = metric->zeta_z;

  float *lam3d = md->lambda;
  float * mu3d = md->mu;

  float *matVx2Vz = bdryfree->matVx2Vz2;
  float *matVy2Vz = bdryfree->matVy2Vz2;
  float *matD = bdryfree->matD;
  
  float A[3][3], B[3][3], C[3][3], D[3][3];
  float AB[3][3], AC[3][3];
  float x_et,y_et,z_et;
  float e_n,e_m,e_nm;
  int n_fd;

  float e11, e12, e13, e21, e22, e23, e31, e32, e33;
  float lam2mu, lam, mu;

  // use local stack array for speedup
  float  lfd_coef[fd_len];
  size_t lfdy_shift[fd_len];
  // put fd op into local array
  for (int i=0; i<fd_len; i++) {
    lfd_coef[i] = fd_coef[i];
    lfdy_shift[i] = fd_indx[i] * siz_iy;
  }
 
  int k = nk2;

  for (int j = nj1; j <= nj2; j++)
  {
    for (int i = ni1; i <= ni2; i++)
    {
      size_t iptr = i + j * siz_iy + k * siz_iz;
      e11 = xi_x[iptr];
      e12 = xi_y[iptr];
      e13 = xi_z[iptr];
      e21 = et_x[iptr];
      e22 = et_y[iptr];
      e23 = et_z[iptr];
      e31 = zt_x[iptr];
      e32 = zt_y[iptr];
      e33 = zt_z[iptr];

      M_FD_SHIFT(x_et, x3d, iptr, fd_len, lfdy_shift, lfd_coef, n_fd);
      M_FD_SHIFT(y_et, y3d, iptr, fd_len, lfdy_shift, lfd_coef, n_fd);
      M_FD_SHIFT(z_et, z3d, iptr, fd_len, lfdy_shift, lfd_coef, n_fd);

      e_n = 1.0/sqrt(x_et*x_et+y_et*y_et+z_et*z_et);
      e_m = 1.0/sqrt(e31*e31+e32*e32+e33*e33);
      e_nm = e_n*e_m;

      lam    = lam3d[iptr];
      mu     =  mu3d[iptr];
      lam2mu = lam + 2.0f * mu;

      // first dim: irow; sec dim: jcol, as Fortran code
      A[0][0] = lam2mu*e31*e31 + mu*(e32*e32+e33*e33);
      A[0][1] = lam*e31*e32 + mu*e32*e31;
      A[0][2] = lam*e31*e33 + mu*e33*e31;
      A[1][0] = lam*e32*e31 + mu*e31*e32;
      A[1][1] = lam2mu*e32*e32 + mu*(e31*e31+e33*e33);
      A[1][2] = lam*e32*e33 + mu*e33*e32;
      A[2][0] = lam*e33*e31 + mu*e31*e33;
      A[2][1] = lam*e33*e32 + mu*e32*e33;
      A[2][2] = lam2mu*e33*e33 + mu*(e31*e31+e32*e32);
      fdlib_math_invert3x3(A);

      B[0][0] = -lam2mu*e31*e11 - mu*(e32*e12+e33*e13);
      B[0][1] = -lam*e31*e12 - mu*e32*e11;
      B[0][2] = -lam*e31*e13 - mu*e33*e11;
      B[1][0] = -lam*e32*e11 - mu*e31*e12;
      B[1][1] = -lam2mu*e32*e12 - mu*(e31*e11+e33*e13);
      B[1][2] = -lam*e32*e13 - mu*e33*e12;
      B[2][0] = -lam*e33*e11 - mu*e31*e13;
      B[2][1] = -lam*e33*e12 - mu*e32*e13;
      B[2][2] = -lam2mu*e33*e13 - mu*(e31*e11+e32*e12);

      C[0][0] = -lam2mu*e31*e21 - mu*(e32*e22+e33*e23);
      C[0][1] = -lam*e31*e22 - mu*e32*e21;
      C[0][2] = -lam*e31*e23 - mu*e33*e21;
      C[1][0] = -lam*e32*e21 - mu*e31*e22;
      C[1][1] = -lam2mu*e32*e22 - mu*(e31*e21+e33*e23);
      C[1][2] = -lam*e32*e23 - mu*e33*e22;
      C[2][0] = -lam*e33*e21 - mu*e31*e23;
      C[2][1] = -lam*e33*e22 - mu*e32*e23;
      C[2][2] = -lam2mu*e33*e23 - mu*(e31*e21+e32*e22);

      fdlib_math_matmul3x3(A, B, AB);
      fdlib_math_matmul3x3(A, C, AC);

      D[0][0] = (y_et*e33-z_et*e32)*e_nm;
      D[0][1] = (z_et*e31-x_et*e33)*e_nm;
      D[0][2] = (x_et*e32-y_et*e31)*e_nm;
      D[1][0] = x_et*e_n;
      D[1][1] = y_et*e_n;
      D[1][2] = z_et*e_n;
      D[2][0] = e31*e_m;
      D[2][1] = e32*e_m;
      D[2][2] = e33*e_m;

      size_t ij = (j * siz_iy + i) * 9;

      // save into mat
      for(int irow = 0; irow < 3; irow++){
        for(int jcol = 0; jcol < 3; jcol++){
          matVx2Vz[ij + irow*3 + jcol] = AB[irow][jcol];
          matVy2Vz[ij + irow*3 + jcol] = AC[irow][jcol];
          matD[ij + irow*3 + jcol] = D[irow][jcol]; 
        }
      }
    }
  }

  return 0;
}
