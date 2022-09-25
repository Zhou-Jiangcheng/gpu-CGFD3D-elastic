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

#include "fd_t.h"
#include "gd_info.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "src_t.h"
#include "bdry_t.h"
/*
 * implement traction image boundary 
 */

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
    size_t siz_line, size_t siz_slice, 
    int fdx_len, int * fdx_indx, float * lfdx_coef,
    int fdy_len, int * fdy_indx, float * lfdy_coef,
    int fdz_len, int * fdz_indx, float * lfdz_coef,
    const int myid, const int verbose)
{
  // loop var for fd
  int n_fd; // loop var for fd

  // local var
  float DxTx,DyTy,DzTz;
  float slwjac;
  float xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz;

  // to save traction and other two dir force var
  float vecxi[5] = {0.0};
  float vecet[5] = {0.0};
  float veczt[5] = {0.0};
  int n, iptr4vec;

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  // last indx, free surface force Tx/Ty/Tz to 0 in cal
  size_t k_min = nk2 - fdz_indx[fdz_len-1];

  // point affected by timg
  for (size_t k=k_min; k <= nk2; k++)
  {
    // k corresponding to 0 index of the fd op

    // index of free surface
    int n_free = nk2 - k - fdz_indx[0]; // first indx is negative

    if(ix<ni && iy<nj)
    {

      size_t iptr = (ix+ni1) + (iy+nj1) * siz_line + k * siz_slice;
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

      // slowness and jac
      slwjac = slw3d[iptr] / jac3d[iptr];

      //
      // for hVx
      //

      // transform to conservative vars
      for (n=0; n<fdx_len; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txx[iptr4vec]
                                      + xi_y[iptr4vec] * Txy[iptr4vec]
                                      + xi_z[iptr4vec] * Txz[iptr4vec] );
      }
      for (n=0; n<fdy_len; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txx[iptr4vec]
                                      + et_y[iptr4vec] * Txy[iptr4vec]
                                      + et_z[iptr4vec] * Txz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n] * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txx[iptr4vec]
                                      + zt_y[iptr4vec] * Txy[iptr4vec]
                                      + zt_z[iptr4vec] * Txz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<fdz_len; n++)
      {
        int n_img = fdz_indx[n] - 2*(n-n_free);
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txx[iptr4vec]
                                       + zt_y[iptr4vec] * Txy[iptr4vec]
                                       + zt_z[iptr4vec] * Txz[iptr4vec] );
      }

      // deri
      M_FD_NOINDX(DxTx, vecxi, fdx_len, lfdx_coef, n_fd);
      M_FD_NOINDX(DyTy, vecet, fdy_len, lfdy_coef, n_fd);
      M_FD_NOINDX(DzTz, veczt, fdz_len, lfdz_coef, n_fd);

      hVx[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;

      //
      // for hVy
      //

      // transform to conservative vars
      for (n=0; n<fdx_len; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txy[iptr4vec]
                                      + xi_y[iptr4vec] * Tyy[iptr4vec]
                                      + xi_z[iptr4vec] * Tyz[iptr4vec] );
      }
      for (n=0; n<fdy_len; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txy[iptr4vec]
                                      + et_y[iptr4vec] * Tyy[iptr4vec]
                                      + et_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n] * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txy[iptr4vec]
                                      + zt_y[iptr4vec] * Tyy[iptr4vec]
                                      + zt_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<fdz_len; n++) {
        int n_img = fdz_indx[n] - 2*(n-n_free);
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txy[iptr4vec]
                                       + zt_y[iptr4vec] * Tyy[iptr4vec]
                                       + zt_z[iptr4vec] * Tyz[iptr4vec] );
      }

      // deri
      M_FD_NOINDX(DxTx, vecxi, fdx_len, lfdx_coef, n_fd);
      M_FD_NOINDX(DyTy, vecet, fdy_len, lfdy_coef, n_fd);
      M_FD_NOINDX(DzTz, veczt, fdz_len, lfdz_coef, n_fd);

      hVy[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;

      //
      // for hVz
      //

      // transform to conservative vars
      for (n=0; n<fdx_len; n++) {
        iptr4vec = iptr + fdx_indx[n];
        vecxi[n] = jac3d[iptr4vec] * (  xi_x[iptr4vec] * Txz[iptr4vec]
                                      + xi_y[iptr4vec] * Tyz[iptr4vec]
                                      + xi_z[iptr4vec] * Tzz[iptr4vec] );
      }
      for (n=0; n<fdy_len; n++) {
        iptr4vec = iptr + fdy_indx[n] * siz_line;
        vecet[n] = jac3d[iptr4vec] * (  et_x[iptr4vec] * Txz[iptr4vec]
                                      + et_y[iptr4vec] * Tyz[iptr4vec]
                                      + et_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // blow surface -> cal
      for (n=0; n<n_free; n++) {
        iptr4vec = iptr + fdz_indx[n] * siz_slice;
        veczt[n] = jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txz[iptr4vec]
                                      + zt_y[iptr4vec] * Tyz[iptr4vec]
                                      + zt_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // at surface -> set to 0
      veczt[n_free] = 0.0;

      // above surface -> mirror
      for (n=n_free+1; n<fdz_len; n++) {
        int n_img = fdz_indx[n] - 2*(n-n_free);
        iptr4vec = iptr + n_img * siz_slice;
        veczt[n] = -jac3d[iptr4vec] * (  zt_x[iptr4vec] * Txz[iptr4vec]
                                       + zt_y[iptr4vec] * Tyz[iptr4vec]
                                       + zt_z[iptr4vec] * Tzz[iptr4vec] );
      }

      // for hVx 
      M_FD_NOINDX(DxTx, vecxi, fdx_len, lfdx_coef, n_fd);
      M_FD_NOINDX(DyTy, vecet, fdy_len, lfdy_coef, n_fd);
      M_FD_NOINDX(DzTz, veczt, fdz_len, lfdz_coef, n_fd);

      hVz[iptr] = ( DxTx+DyTy+DzTz ) * slwjac;
    }
  }

  return;
}

/*******************************************************************************
 * add source terms
 ******************************************************************************/

__global__ void
sv_curv_col_el_iso_rhs_src_gpu(
    float * hVx , float * hVy , float * hVz ,
    float * hTxx, float * hTyy, float * hTzz,
    float * hTxz, float * hTyz, float * hTxy,
    float * jac3d, float * slw3d,
    src_t src, // short nation for reference member
    const int myid, const int verbose)
{
  // for easy coding and efficiency
  int max_ext = src.max_ext;

  // get fi / mij
  float fx, fy, fz;
  float Mxx,Myy,Mzz,Mxz,Myz,Mxy;

  int it     = src.it;
  int istage = src.istage;
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;

  // add src; ix is a commont iterater var
  if(ix<src.total_number)
  {
    int   it_start = src.it_begin[ix];
    int   it_end   = src.it_end  [ix];

    if (it >= it_start && it <= it_end)
    {
      int   *ptr_ext_indx = src.ext_indx + ix * max_ext;
      float *ptr_ext_coef = src.ext_coef + ix * max_ext;
      int it_to_it_start = it - it_start;
      size_t iptr_cur_stage =   ix * src.max_nt * src.max_stage // skip other src
                           + it_to_it_start * src.max_stage // skip other time step
                           + istage;
      if (src.force_actived == 1) {
        fx  = src.Fx [iptr_cur_stage];
        fy  = src.Fy [iptr_cur_stage];
        fz  = src.Fz [iptr_cur_stage];
      }
      if (src.moment_actived == 1) {
        Mxx = src.Mxx[iptr_cur_stage];
        Myy = src.Myy[iptr_cur_stage];
        Mzz = src.Mzz[iptr_cur_stage];
        Mxz = src.Mxz[iptr_cur_stage];
        Myz = src.Myz[iptr_cur_stage];
        Mxy = src.Mxy[iptr_cur_stage];
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
          atomicAdd(&hTxx[iptr], -Mxx * rjac);
          atomicAdd(&hTyy[iptr], -Myy * rjac);
          atomicAdd(&hTzz[iptr], -Mzz * rjac);
          atomicAdd(&hTxz[iptr], -Mxz * rjac);
          atomicAdd(&hTyz[iptr], -Myz * rjac);
          atomicAdd(&hTxy[iptr], -Mxy * rjac);
        }
      } // i_ext
    } // it
  }

  return;
}

int
sv_eq1st_curv_graves_Qs(float *w, int ncmp, float dt, gdinfo_t *gdinfo, md_t *md)
{
  int ierr = 0;

  float coef = - PI * md->visco_Qs_freq * dt;

  for (int icmp=0; icmp<ncmp; icmp++)
  {
    float *var = w + icmp * gdinfo->siz_icmp;

    for (int k = gdinfo->nk1; k <= gdinfo->nk2; k++)
    {
      for (int j = gdinfo->nj1; j <= gdinfo->nj2; j++)
      {
        for (int i = gdinfo->ni1; i <= gdinfo->ni2; i++)
        {
          size_t iptr = i + j * gdinfo->siz_iy + k * gdinfo->siz_iz;

          float Qatt = expf( coef / md->Qs[iptr] );

          var[iptr] *= Qatt;
        }
      }
    }
  }

  return ierr;
}
