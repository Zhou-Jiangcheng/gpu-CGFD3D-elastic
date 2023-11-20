/*********************************************************************
 * wavefield for 3d elastic 1st-order equations
 **********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "fdlib_mem.h"
#include "wav_t.h"

int 
wav_init(gd_t *gd,
         wav_t *V,
         int number_of_levels)
{
  int ierr = 0;

  V->nx   = gd->nx;
  V->ny   = gd->ny;
  V->nz   = gd->nz;
  V->ncmp = 9;
  V->nlevel = number_of_levels;

  V->siz_iy   = V->nx;
  V->siz_iz   = V->nx * V->ny;
  V->siz_icmp = V->nx * V->ny * V->nz;
  V->siz_ilevel = V->siz_icmp * V->ncmp;

  // vars
  // 3 Vi, 6 Tij, 4 rk stages
  //V->v5d = (float *) fdlib_mem_calloc_1d_float(V->siz_ilevel * V->nlevel,
  //                      0.0, "v5d, wf_el3d_1st");
  // no need alloc space in cpu, all wave in device.
  // just need alloc space to recv wave from device
  // V->siz_icmp * (V->ncmp+3) is the max limit
  // Vx, Vy, Vz, Txx, Tyy, Tzz, Tyz, Txz, Txy, Exx, Eyy, Ezz, Eyz, Exz, Exy
  V->v5d = (float *) fdlib_mem_calloc_1d_float(V->siz_icmp * (V->ncmp+6),
                        0.0, "v5d, wf_el3d_1st");
  // position of each var
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(
                      V->ncmp, 0, "w3d_pos, wf_el3d_1st");
  // name of each var
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(
                      V->ncmp, CONST_MAX_STRLEN, "w3d_name, wf_el3d_1st");
  
  // set value
  for (int icmp=0; icmp < V->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * V->siz_icmp;
  }

  // set values
  int icmp = 0;

  /*
   * 0-3: Vx,Vy,Vz
   * 4-9: Txx,Tyy,Tzz,Txz,Tyz,Txy
   */

  sprintf(cmp_name[icmp],"%s","Vx");
  V->Vx_pos = cmp_pos[icmp];
  V->Vx_seq = 0;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vy");
  V->Vy_pos = cmp_pos[icmp];
  V->Vy_seq = 1;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vz");
  V->Vz_pos = cmp_pos[icmp];
  V->Vz_seq = 2;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Txx");
  V->Txx_pos = cmp_pos[icmp];
  V->Txx_seq = 3;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Tyy");
  V->Tyy_pos = cmp_pos[icmp];
  V->Tyy_seq = 4;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Tzz");
  V->Tzz_pos = cmp_pos[icmp];
  V->Tzz_seq = 5;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Tyz");
  V->Tyz_pos = cmp_pos[icmp];
  V->Tyz_seq = 6;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Txz");
  V->Txz_pos = cmp_pos[icmp];
  V->Txz_seq = 7;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Txy");
  V->Txy_pos = cmp_pos[icmp];
  V->Txy_seq = 8;
  icmp += 1;

  // set pointer
  V->cmp_pos  = cmp_pos;
  V->cmp_name = cmp_name;

  return ierr;
}

int 
wav_ac_init(gd_t *gd,
            wav_t *V,
            int number_of_levels)
{
  int ierr = 0;

  // Vx,Vy,Vz,P
  V->ncmp = 4;

  V->nx   = gd->nx;
  V->ny   = gd->ny;
  V->nz   = gd->nz;
  V->nlevel = number_of_levels;

  V->siz_iy   = V->nx;
  V->siz_iz   = V->nx * V->ny;
  V->siz_icmp = V->nx * V->ny * V->nz;
  V->siz_ilevel = V->siz_icmp * V->ncmp;

  // vars
  // 3 Vi, 6 Tij, 4 rk stages
  V->v5d = (float *) fdlib_mem_calloc_1d_float(V->siz_ilevel * V->nlevel,
                        0.0, "v5d, wf_ac3d_1st");
  // position of each var
  size_t *cmp_pos = (size_t *) fdlib_mem_calloc_1d_sizet(
                      V->ncmp, 0, "w3d_pos, wf_ac3d_1st");
  // name of each var
  char **cmp_name = (char **) fdlib_mem_malloc_2l_char(
                      V->ncmp, CONST_MAX_STRLEN, "w3d_name, wf_ac3d_1st");
  
  // set value
  for (int icmp=0; icmp < V->ncmp; icmp++)
  {
    cmp_pos[icmp] = icmp * V->siz_icmp;
  }

  // set values
  int icmp = 0;

  /*
   * 0-3: Vx,Vy,Vz
   * 4: P
   */

  sprintf(cmp_name[icmp],"%s","Vx");
  V->Vx_pos = cmp_pos[icmp];
  V->Vx_seq = 0;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vy");
  V->Vy_pos = cmp_pos[icmp];
  V->Vy_seq = 1;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","Vz");
  V->Vz_pos = cmp_pos[icmp];
  V->Vz_seq = 2;
  icmp += 1;

  sprintf(cmp_name[icmp],"%s","P");
  V->Txx_pos = cmp_pos[icmp];
  V->Txx_seq = 3;
  icmp += 1;

  // set pointer
  V->cmp_pos  = cmp_pos;
  V->cmp_name = cmp_name;

  return ierr;
}

int
wav_check_value(float *w, wav_t *wav)
{
  int ierr = 0;

  for (int icmp=0; icmp < wav->ncmp; icmp++)
  {
    float *ptr = w + icmp * wav->siz_icmp;
    for (size_t iptr=0; iptr < wav->siz_icmp; iptr++)
    {
      if (ptr[iptr] != ptr[iptr])
      {
        fprintf(stderr, "ERROR: NaN occurs at iptr=%d icmp=%d\n", iptr, icmp);
        fflush(stderr);
        exit(-1);
      }
    }
  }

  return ierr;
}

__global__ void
PG_calcu_gpu(float *w_end, float *w_pre, gd_t gd_d, float *PG, float *Dis_accu, float dt)
{
  //Dis_accu is displacement accumulation.
  int ni = gd_d.ni;
  int nj = gd_d.nj;
  int ni1 = gd_d.ni1;
  int nj1 = gd_d.nj1;
  int nk1 = gd_d.nk1;
  int ni2 = gd_d.ni2;
  int nj2 = gd_d.nj2;
  int nk2 = gd_d.nk2;
  size_t siz_iy = gd_d.siz_iy;
  size_t siz_iz = gd_d.siz_iz;
  size_t siz_icmp  = gd_d.siz_icmp;
  // 0-2 Vx1,Vy1,Vz1  it+1 moment V
  // 0-2 Vx0,Vy0,Vz0  it moment V
  float *Vx1 = w_end + 0*siz_icmp;
  float *Vy1 = w_end + 1*siz_icmp;
  float *Vz1 = w_end + 2*siz_icmp;
  float *Vx0 = w_pre + 0*siz_icmp;
  float *Vy0 = w_pre + 1*siz_icmp;
  float *Vz0 = w_pre + 2*siz_icmp;
  float *PGV  = PG + 0 *siz_iz;
  float *PGVh = PG + 1 *siz_iz;
  float *PGVx = PG + 2 *siz_iz;
  float *PGVy = PG + 3 *siz_iz;
  float *PGVz = PG + 4 *siz_iz;
  float *PGA  = PG + 5 *siz_iz;
  float *PGAh = PG + 6 *siz_iz;
  float *PGAx = PG + 7 *siz_iz;
  float *PGAy = PG + 8 *siz_iz;
  float *PGAz = PG + 9 *siz_iz;
  float *PGD  = PG + 10*siz_iz;
  float *PGDh = PG + 11*siz_iz;
  float *PGDx = PG + 12*siz_iz;
  float *PGDy = PG + 13*siz_iz;
  float *PGDz = PG + 14*siz_iz;
  float *D_x = Dis_accu + 0*siz_iz;
  float *D_y = Dis_accu + 1*siz_iz;
  float *D_z = Dis_accu + 2*siz_iz;
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iptr,iptr1;

  if(ix<ni && iy<nj)
  {
    iptr  = (ix+ni1) + (iy+nj1) * siz_iy + nk2 * siz_iz;
    iptr1 = (ix+ni1) + (iy+nj1) * siz_iy;
    float V, Vh, D, Dh, A, Ah, Ax, Ay, Az;
    Ax = fabs((Vx1[iptr]-Vx0[iptr])/dt);
    Ay = fabs((Vy1[iptr]-Vy0[iptr])/dt);
    Az = fabs((Vz1[iptr]-Vz0[iptr])/dt);
    D_x[iptr1] += 0.5*(Vx1[iptr]+Vx0[iptr])*dt;
    D_y[iptr1] += 0.5*(Vy1[iptr]+Vy0[iptr])*dt;
    D_z[iptr1] += 0.5*(Vz1[iptr]+Vz0[iptr])*dt;
    V  = sqrt(Vx1[iptr]*Vx1[iptr] + Vy1[iptr]*Vy1[iptr] + Vz1[iptr]*Vz1[iptr]);
    Vh = sqrt(Vx1[iptr]*Vx1[iptr] + Vy1[iptr]*Vy1[iptr]);
    A  = sqrt(Ax*Ax + Ay*Ay + Az*Az);
    Ah = sqrt(Ax*Ax + Ay*Ay);
    D  = sqrt(D_x[iptr1]*D_x[iptr1] + D_y[iptr1]*D_y[iptr1] + D_z[iptr1]*D_z[iptr1]);
    Dh = sqrt(D_x[iptr1]*D_x[iptr1] + D_y[iptr1]*D_y[iptr1]);

    if(PGVx[iptr1] < fabs(Vx1[iptr]))   PGVx[iptr1]=fabs(Vx1[iptr]);
    if(PGVy[iptr1] < fabs(Vy1[iptr]))   PGVy[iptr1]=fabs(Vy1[iptr]);
    if(PGVz[iptr1] < fabs(Vz1[iptr]))   PGVz[iptr1]=fabs(Vz1[iptr]);
    if(PGAx[iptr1] < Ax)                PGAx[iptr1]=Ax;
    if(PGAy[iptr1] < Ay)                PGAy[iptr1]=Ay;
    if(PGAz[iptr1] < Az)                PGAz[iptr1]=Az;
    if(PGDx[iptr1] < fabs(D_x[iptr1]))  PGDx[iptr1]=abs(D_x[iptr1]);
    if(PGDy[iptr1] < fabs(D_y[iptr1]))  PGDy[iptr1]=abs(D_y[iptr1]);
    if(PGDz[iptr1] < fabs(D_z[iptr1]))  PGDz[iptr1]=abs(D_z[iptr1]);
    if(PGV[iptr1]  < V)                 PGV[iptr1]=V;
    if(PGVh[iptr1] < Vh)                PGVh[iptr1]=Vh;
    if(PGA[iptr1]  < A)                 PGA[iptr1]=A;
    if(PGAh[iptr1] < Ah)                PGAh[iptr1]=Ah;
    if(PGD[iptr1]  < D)                 PGD[iptr1]=D;
    if(PGDh[iptr1] < Dh)                PGDh[iptr1]=Dh;
  }
}

__global__ void
wav_update(size_t size, float coef, float *w_update, float *w_input1, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if(ix<size){
    w_update[ix] = w_input1[ix] + coef * w_input2[ix];
  }
}

__global__ void
wav_update_end(size_t size, float coef, float *w_update, float *w_input2)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if(ix<size){
    w_update[ix] = w_update[ix] + coef * w_input2[ix];
  }
}

