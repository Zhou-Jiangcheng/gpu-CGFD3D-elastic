// todo:
//  check : abs_set_ablexp
//  convert fortrn to c: abs_ablexp_cal_damp

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "fdlib_math.h"
#include "fdlib_mem.h"
#include "bdry_t.h"

//- may move to par file
#define CONSPD 2.0f // power for d
#define CONSPB 2.0f // power for beta
#define CONSPA 1.0f // power for alpha

/*
 *   init bdry_t
 */

int
bdry_init(bdry_t *bdry, int nx, int ny, int nz)
{
  bdry->is_enable_pml  = 0;
  bdry->is_enable_mpml = 0;
  bdry->is_enable_ablexp  = 0;
  bdry->is_enable_free = 0;

  bdry->nx = nx;
  bdry->ny = ny;
  bdry->nz = nz;

  return 0;
}

/*
 * matrix for velocity gradient conversion
 *  only implement z2 (top) right now
 */

int
bdry_free_set(gdinfo_t    *gdinfo,
              bdry_t      *bdryfree,
              int   *neighid, 
              int   in_is_sides[][2],
              const int verbose)
{
  int ierr = 0;

  size_t siz_slice  = gdinfo->siz_iz;

  // default disable
  bdryfree->is_enable_free = 0;

  // check each side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      int ind_1d = iside + idim * 2;

      bdryfree->is_sides_free  [idim][iside] = in_is_sides[idim][iside];

      // reset 0 if not mpi boundary
      if (neighid[ind_1d] != MPI_PROC_NULL)
      {
        bdryfree->is_sides_free  [idim][iside] = 0;
      }

      // enable if any side valid
      if (bdryfree->is_sides_free  [idim][iside] == 1) {
        bdryfree->is_enable_free = 1;
      }
    } // iside
  } // idim

  // following only implement z2 (top) right now
  float *matVx2Vz = (float *)fdlib_mem_calloc_1d_float(
                                      siz_slice * CONST_NDIM * CONST_NDIM,
                                      0.0,
                                      "bdry_free_set");

  float *matVy2Vz = (float *)fdlib_mem_calloc_1d_float(
                                      siz_slice * CONST_NDIM * CONST_NDIM,
                                      0.0,
                                      "bdry_free_set");

  bdryfree->matVx2Vz2 = matVx2Vz;
  bdryfree->matVy2Vz2 = matVy2Vz;

  return ierr;
}

/*
 * set up abs_coefs for cfs-pml
 */
void
bdry_pml_set(gdinfo_t *gdinfo,
             gd_t *gd,
             wav_t *wav,
             bdry_t *bdrypml,
             int   *neighid, 
             int   in_is_sides[][2],
             int   in_num_layers[][2],
             float in_alpha_max[][2], //
             float in_beta_max[][2], //
             float in_velocity[][2], //
             int verbose)
{
  int    ni1 = gdinfo->ni1;
  int    ni2 = gdinfo->ni2;
  int    nj1 = gdinfo->nj1;
  int    nj2 = gdinfo->nj2;
  int    nk1 = gdinfo->nk1;
  int    nk2 = gdinfo->nk2;
  int    nx  = gdinfo->nx ;
  int    ny  = gdinfo->ny ;
  int    nz  = gdinfo->nz ;
  int    siz_line = gdinfo->siz_iy;
  int    siz_slice = gdinfo->siz_iz;

  // default disable
  bdrypml->is_enable_pml = 0;

  // check each side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      int ind_1d = iside + idim * 2;

      // default set to input
      bdrypml->is_sides_pml  [idim][iside] = in_is_sides[idim][iside];
      bdrypml->num_of_layers[idim][iside] = in_num_layers[idim][iside];

      // reset 0 if not mpi boundary
      if (neighid[ind_1d] != MPI_PROC_NULL)
      {
        bdrypml->is_sides_pml  [idim][iside] = 0;
        bdrypml->num_of_layers[idim][iside] = 0;
      }

      // default loop index
      bdrypml->ni1[idim][iside] = ni1;
      bdrypml->ni2[idim][iside] = ni2;
      bdrypml->nj1[idim][iside] = nj1;
      bdrypml->nj2[idim][iside] = nj2;
      bdrypml->nk1[idim][iside] = nk1;
      bdrypml->nk2[idim][iside] = nk2;

      // shrink to actual size
      if (idim == 0 && iside ==0) { // x1
        bdrypml->ni2[idim][iside] = ni1 + bdrypml->num_of_layers[idim][iside];
      }
      if (idim == 0 && iside ==1) { // x2
        bdrypml->ni1[idim][iside] = ni2 - bdrypml->num_of_layers[idim][iside];
      }
      if (idim == 1 && iside ==0) { // y1
        bdrypml->nj2[idim][iside] = nj1 + bdrypml->num_of_layers[idim][iside];
      }
      if (idim == 1 && iside ==1) { // y2
        bdrypml->nj1[idim][iside] = nj2 - bdrypml->num_of_layers[idim][iside];
      }
      if (idim == 2 && iside ==0) { // z1
        bdrypml->nk2[idim][iside] = nk1 + bdrypml->num_of_layers[idim][iside];
      }
      if (idim == 2 && iside ==1) { // z2
        bdrypml->nk1[idim][iside] = nk2 - bdrypml->num_of_layers[idim][iside];
      }

      // enable if any side valid
      if (bdrypml->is_sides_pml  [idim][iside] == 1) {
        bdrypml->is_enable_pml = 1;
      }

    } // iside
  } // idim

  // alloc coef
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      if (bdrypml->is_sides_pml[idim][iside] == 1) {
        int npoints = bdrypml->num_of_layers[idim][iside] + 1;
        bdrypml->A[idim][iside] = (float *)malloc( npoints * sizeof(float));
        bdrypml->B[idim][iside] = (float *)malloc( npoints * sizeof(float));
        bdrypml->D[idim][iside] = (float *)malloc( npoints * sizeof(float));
      } else {
        bdrypml->A[idim][iside] = NULL;
        bdrypml->B[idim][iside] = NULL;
        bdrypml->D[idim][iside] = NULL;
      }
    }
  }

  // cal coef for each dim and side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      // skip if not pml
      if (bdrypml->is_sides_pml[idim][iside] == 0) continue;

      float *A = bdrypml->A[idim][iside];
      float *B = bdrypml->B[idim][iside];
      float *D = bdrypml->D[idim][iside];

      // esti L0 and dh
      float L0, dh;
      bdry_cal_abl_len_dh(gd,bdrypml->ni1[idim][iside],
                             bdrypml->ni2[idim][iside],
                             bdrypml->nj1[idim][iside],
                             bdrypml->nj2[idim][iside],
                             bdrypml->nk1[idim][iside],
                             bdrypml->nk2[idim][iside],
                             idim,
                             &L0, &dh);

      // para
      int npoints = bdrypml->num_of_layers[idim][iside] + 1;
      float num_lay = npoints - 1;
      float Rpp  = bdry_pml_cal_R(num_lay);
      float dmax = bdry_pml_cal_dmax(L0, in_velocity[idim][iside], Rpp);
      float amax = in_alpha_max[idim][iside];
      float bmax = in_beta_max[idim][iside];

      // from PML-interior to outer side
      for (int n=0; n<npoints; n++)
      {
        // first point has non-zero value
        float L = n * dh;
        int i;

        // convert to grid index from left to right
        if (iside == 0) { // x1/y1/z1
          i = npoints - 1 - n;
        } else { // x2/y2/z2
          i = n; 
        }

        D[i] = bdry_pml_cal_d( L, L0, dmax );
        A[i] = bdry_pml_cal_a( L, L0, amax );
        B[i] = bdry_pml_cal_b( L, L0, bmax );

        // convert d_x to d_x/beta_x since only d_x/beta_x needed
        D[i] /= B[i];
        // covert ax = a_x + d_x/beta_x 
        A[i] += D[i];
        // covert bx = 1.0/bx 
        B[i] = 1.0 / B[i];
      }

    } // iside
  } // idim

  // alloc auxvar
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      int nx = (bdrypml->ni2[idim][iside] - bdrypml->ni1[idim][iside] + 1);
      int ny = (bdrypml->nj2[idim][iside] - bdrypml->nj1[idim][iside] + 1);
      int nz = (bdrypml->nk2[idim][iside] - bdrypml->nk1[idim][iside] + 1);

      bdry_pml_auxvar_init(nx,ny,nz,wav,
                           &(bdrypml->auxvar[idim][iside]),verbose);
    } // iside
  } // idim

}

// alloc auxvar
void
bdry_pml_auxvar_init(int nx, int ny, int nz, 
                     wav_t *wav,
                     bdrypml_auxvar_t *auxvar,
                     const int verbose)
{
  auxvar->nx   = nx;
  auxvar->ny   = ny;
  auxvar->nz   = nz;
  auxvar->ncmp = wav->ncmp;
  auxvar->nlevel = wav->nlevel;

  auxvar->siz_iy   = auxvar->nx;
  auxvar->siz_iz   = auxvar->nx * auxvar->ny;
  auxvar->siz_icmp = auxvar->nx * auxvar->ny * auxvar->nz;
  auxvar->siz_ilevel = auxvar->siz_icmp * auxvar->ncmp;

  auxvar->Vx_pos  = wav->Vx_seq  * auxvar->siz_icmp;
  auxvar->Vy_pos  = wav->Vy_seq  * auxvar->siz_icmp;
  auxvar->Vz_pos  = wav->Vz_seq  * auxvar->siz_icmp;
  auxvar->Txx_pos = wav->Txx_seq * auxvar->siz_icmp;
  auxvar->Tyy_pos = wav->Tyy_seq * auxvar->siz_icmp;
  auxvar->Tzz_pos = wav->Tzz_seq * auxvar->siz_icmp;
  auxvar->Tyz_pos = wav->Tyz_seq * auxvar->siz_icmp;
  auxvar->Txz_pos = wav->Txz_seq * auxvar->siz_icmp;
  auxvar->Txy_pos = wav->Txy_seq * auxvar->siz_icmp;

  // vars
  // contain all vars at each side, include rk scheme 4 levels vars
  if (auxvar->siz_icmp > 0 ) { // valid pml layer
    auxvar->var = (float *) fdlib_mem_calloc_1d_float( 
                 auxvar->siz_ilevel * auxvar->nlevel,
                 0.0, "bdry_pml_auxvar_init");
  } else { // nx,ny,nz has 0
    auxvar->var = NULL;
  }
}

float
bdry_pml_cal_R(float num_lay)
{
  // use corrected Rpp
  return (float) (pow(10, -( (log10((double)num_lay)-1.0)/log10(2.0) + 4.0)));
}

float
bdry_pml_cal_dmax(float L, float Vp, float Rpp)
{
  return (float) (-Vp / (2.0 * L) * log(Rpp) * (CONSPD + 1.0));
}

float
bdry_pml_cal_amax(float fc)
{return PI*fc;}

float
bdry_pml_cal_d(float x, float L, float dmax)
{
  return (x<0) ? 0.0f : (float) (dmax * pow(x/L, CONSPD));
}

float
bdry_pml_cal_a(float x, float L, float amax)
{
  return (x<0) ? 0.0f : (float) (amax * (1.0 - pow(x/L, CONSPA)));
}

float
bdry_pml_cal_b(float x, float L, float bmax)
{
  return (x<0) ? 1.0f : (float) (1.0 + (bmax-1.0) * pow(x/L, CONSPB));
}

//---------------------------------------------
//esti L and dh along idim damping layers
int
bdry_cal_abl_len_dh(gd_t *gd, 
                    int abs_ni1, int abs_ni2,
                    int abs_nj1, int abs_nj2,
                    int abs_nk1, int abs_nk2,
                    int idim,
                    float *avg_L, float *avg_dh)
{
  int ierr = 0;

  int siz_line  = gd->siz_iy;
  int siz_slice = gd->siz_iz;

  // cartesian grid is simple
  if (gd->type == GD_TYPE_CART)
  {
    if (idim == 0) { // x-axis
      *avg_dh = gd->dx;
      *avg_L  = gd->dx * (abs_ni2 - abs_ni1);
    } else if (idim == 1) { // y-axis
      *avg_dh = gd->dy;
      *avg_L  = gd->dy * (abs_nj2 - abs_nj1);
    } else { // z-axis
      *avg_dh = gd->dz;
      *avg_L  = gd->dz * (abs_nk2 - abs_nk1);
    }
  }
  // curv grid needs avg
  else if (gd->type == GD_TYPE_CURV)
  {
    float *x3d = gd->x3d;
    float *y3d = gd->y3d;
    float *z3d = gd->z3d;

    double L  = 0.0;
    double dh = 0.0;
    int    num = 0;

    if (idim == 0) // x-axis
    {
      for (int k=abs_nk1; k<=abs_nk2; k++)
      {
        for (int j=abs_nj1; j<=abs_nj2; j++)
        {
          int iptr = abs_ni1 + j * siz_line + k * siz_slice;
          double x0 = x3d[iptr];
          double y0 = y3d[iptr];
          double z0 = z3d[iptr];
          for (int i=abs_ni1+1; i<=abs_ni2; i++)
          {
            int iptr = i + j * siz_line + k * siz_slice;

            double x1 = x3d[iptr];
            double y1 = y3d[iptr];
            double z1 = z3d[iptr];

            L += sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );

            x0 = x1;
            y0 = y1;
            z0 = z1;
            num += 1;
          }
        }
      }

      *avg_dh = (float)( L / num );
      *avg_L = (*avg_dh) * (abs_ni2 - abs_ni1);
    } 
    else if (idim == 1) // y-axis
    { 
      for (int k=abs_nk1; k<=abs_nk2; k++)
      {
        for (int i=abs_ni1; i<=abs_ni2; i++)
        {
          int iptr = i + abs_nj1 * siz_line + k * siz_slice;
          double x0 = x3d[iptr];
          double y0 = y3d[iptr];
          double z0 = z3d[iptr];
          for (int j=abs_nj1+1; j<=abs_nj2; j++)
          {
            int iptr = i + j * siz_line + k * siz_slice;

            double x1 = x3d[iptr];
            double y1 = y3d[iptr];
            double z1 = z3d[iptr];

            L += sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );

            x0 = x1;
            y0 = y1;
            z0 = z1;
            num += 1;
          }
        }
      }

      *avg_dh = (float)( L / num );
      *avg_L = (*avg_dh) * (abs_nj2 - abs_nj1);
    }
    else // z-axis
    { 
      for (int j=abs_nj1; j<=abs_nj2; j++)
      {
        for (int i=abs_ni1; i<=abs_ni2; i++)
        {
          int iptr = i + j * siz_line + abs_nk1 * siz_slice;
          double x0 = x3d[iptr];
          double y0 = y3d[iptr];
          double z0 = z3d[iptr];
          for (int k=abs_nk1+1; k<=abs_nk2; k++)
          {
            int iptr = i + j * siz_line + k * siz_slice;

            double x1 = x3d[iptr];
            double y1 = y3d[iptr];
            double z1 = z3d[iptr];

            L += sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );

            x0 = x1;
            y0 = y1;
            z0 = z1;
            num += 1;
          }
        }
      }

      *avg_dh = (float)( L / num );
      *avg_L = (*avg_dh) * (abs_nk2 - abs_nk1);
    } // idim

  } // gd type

  return ierr;
}


/*
 * setup ablexp parameters
 */

int
bdry_ablexp_set(gdinfo_t *gdinfo,
                gd_t *gd,
                wav_t *wav,
                bdry_t *bdryexp,
                int   *neighid, 
                int   in_is_sides[][2],
                int   in_num_layers[][2],
                float in_velocity[][2], //
                float dt,
                int  *topoid,
                int verbose)
{
  int ierr = 0;
  int ni1 = gdinfo->ni1;
  int ni2 = gdinfo->ni2;
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  int ni  = gdinfo->ni ;
  int nj  = gdinfo->nj ;
  int nk  = gdinfo->nk ;
  int nx  = gdinfo->nx ;
  int ny  = gdinfo->ny ;
  int nz  = gdinfo->nz ;
  int siz_line = gdinfo->siz_iy;
  int siz_slice = gdinfo->siz_iz;
  int abs_number[CONST_NDIM][2];
  int n;

  // default disable
  bdryexp->is_enable_ablexp = 0;

  // check each side
  for (int idim=0; idim<CONST_NDIM; idim++)
  {
    for (int iside=0; iside<2; iside++)
    {
      int ind_1d = iside + idim * 2;

      // default set to input
      bdryexp->is_sides_ablexp[idim][iside] = in_is_sides[idim][iside];
      bdryexp->num_of_layers[idim][iside] = in_num_layers[idim][iside];

      // reset 0 if not mpi boundary
      if (neighid[ind_1d] != MPI_PROC_NULL)
      {
        bdryexp->is_sides_ablexp  [idim][iside] = 0;
        bdryexp->num_of_layers[idim][iside] = 0;
      }

      // enable if any side valid
      if (bdryexp->is_sides_ablexp  [idim][iside] == 1)
      {
        bdryexp->is_enable_ablexp = 1;
      }
    } // iside
  } // idim

  // block index for ablexp, default inactive
  bdry_block_t *D = bdryexp->bdry_blk;
  for (n=0; n < CONST_NDIM_2; n++)
  {
     D[n].enable = 0;
     D[n].ni1 =  0;
     D[n].ni2 = -1;
     D[n].ni  =  0;

     D[n].nj1 =  0;
     D[n].nj2 = -1;
     D[n].nj  =  0;

     D[n].nk1 =  0;
     D[n].nk2 = -1;
     D[n].nk  =  0;
  }

  // alloc coef
  bdryexp->ablexp_Ex = (float *)malloc( nx * sizeof(float));
  bdryexp->ablexp_Ey = (float *)malloc( ny * sizeof(float));
  bdryexp->ablexp_Ez = (float *)malloc( nz * sizeof(float));
  for (int i=0; i<nx; i++) bdryexp->ablexp_Ex[i] = 1.0;
  for (int j=0; j<ny; j++) bdryexp->ablexp_Ey[j] = 1.0;
  for (int k=0; k<nz; k++) bdryexp->ablexp_Ez[k] = 1.0;

  // x1
  n=0;
  D[n].ni=bdryexp->num_of_layers[0][0]; D[n].ni1=ni1; D[n].ni2=D[n].ni1+D[n].ni-1; 
  D[n].nj=nj                          ; D[n].nj1=nj1; D[n].nj2=D[n].nj1+D[n].nj-1; 
  D[n].nk=nk                          ; D[n].nk1=nk1; D[n].nk2=D[n].nk1+D[n].nk-1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;

     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            0,
                            &L0, &dh);

     for (int i=D[n].ni1; i<=D[n].ni2; i++)
     {
        // the first point of layer is the first dh damping
        bdryexp->ablexp_Ex[i] = bdry_ablexp_cal_mask(D[n].ni - (i - D[n].ni1),
                                                     in_velocity[0][0], dt,
                                                     D[n].ni, dh);
     }
  }

  // x2
  n += 1;
  D[n].ni=bdryexp->num_of_layers[0][1]; D[n].ni1=ni2 - D[n].ni + 1; D[n].ni2=D[n].ni1+D[n].ni-1; 
  D[n].nj=nj                          ; D[n].nj1=nj1              ; D[n].nj2=D[n].nj1+D[n].nj-1; 
  D[n].nk=nk                          ; D[n].nk1=nk1              ; D[n].nk2=D[n].nk1+D[n].nk-1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;

     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            0,
                            &L0, &dh);

     for (int i=D[n].ni1; i<=D[n].ni2; i++)
     {
        bdryexp->ablexp_Ex[i] = bdry_ablexp_cal_mask(i - D[n].ni1 + 1,
                                                     in_velocity[0][1], dt,
                                                     D[n].ni, dh);
     }
  }

  int ni_x = bdryexp->num_of_layers[0][0] + bdryexp->num_of_layers[0][1];
  int nj_y = bdryexp->num_of_layers[1][0] + bdryexp->num_of_layers[1][1];

  //y1
  n += 1;
  D[n].ni = ni - ni_x;
  D[n].ni1= ni1 + bdryexp->num_of_layers[0][0];
  D[n].ni2= D[n].ni1 + D[n].ni - 1; 
  D[n].nj = bdryexp->num_of_layers[1][0];
  D[n].nj1= nj1;
  D[n].nj2= D[n].nj1 + D[n].nj - 1; 
  D[n].nk = nk;
  D[n].nk1= nk1;
  D[n].nk2= D[n].nk1 + D[n].nk - 1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;
     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            1,
                            &L0, &dh);

     for (int j=D[n].nj1; j<=D[n].nj2; j++)
     {
        bdryexp->ablexp_Ey[j] = bdry_ablexp_cal_mask(D[n].nj - (j - D[n].nj1),
                                                     in_velocity[1][0], dt,
                                                     D[n].nj, dh);
     }
  }

  // y2
  n += 1;
  D[n].ni = ni - ni_x;
  D[n].ni1= ni1 + bdryexp->num_of_layers[0][0];
  D[n].ni2= D[n].ni1 + D[n].ni - 1; 
  D[n].nj = bdryexp->num_of_layers[1][1];
  D[n].nj1= nj2 - D[n].nj + 1;
  D[n].nj2= D[n].nj1 + D[n].nj - 1; 
  D[n].nk = nk;
  D[n].nk1= nk1;
  D[n].nk2= D[n].nk1 + D[n].nk - 1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;
     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            1,
                            &L0, &dh);

     for (int j=D[n].nj1; j<=D[n].nj2; j++)
     {
        bdryexp->ablexp_Ey[j] = bdry_ablexp_cal_mask(j - D[n].nj1 + 1,
                                                     in_velocity[1][1], dt,
                                                     D[n].nj, dh);
     }
  }

  // z1
  n += 1;
  D[n].ni = ni - ni_x;
  D[n].ni1= ni1 + bdryexp->num_of_layers[0][0];
  D[n].ni2= D[n].ni1 + D[n].ni - 1; 
  D[n].nj = nj - nj_y;
  D[n].nj1= nj1 + bdryexp->num_of_layers[1][0];
  D[n].nj2= D[n].nj1 + D[n].nj - 1; 
  D[n].nk = bdryexp->num_of_layers[2][0];
  D[n].nk1= nk1;
  D[n].nk2= D[n].nk1 + D[n].nk - 1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;
     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            2,
                            &L0, &dh);

     for (int k=D[n].nk1; k<=D[n].nk2; k++)
     {
        bdryexp->ablexp_Ez[k] = bdry_ablexp_cal_mask(D[n].nk - (k - D[n].nk1),
                                                  in_velocity[2][0], dt,
                                                  D[n].nk, dh);
     }
  }

  // z2
  n += 1;
  D[n].ni = ni - ni_x;
  D[n].ni1= ni1 + bdryexp->num_of_layers[0][0];
  D[n].ni2= D[n].ni1 + D[n].ni - 1; 
  D[n].nj = nj - nj_y;
  D[n].nj1= nj1 + bdryexp->num_of_layers[1][0];
  D[n].nj2= D[n].nj1 + D[n].nj - 1; 
  D[n].nk = bdryexp->num_of_layers[2][1];
  D[n].nk1= nk2 - D[n].nk + 1;
  D[n].nk2= D[n].nk1 + D[n].nk - 1; 
  if (D[n].ni>0 && D[n].nj>0 && D[n].nk>0)
  {
     D[n].enable = 1;
     // esti L0 and dh
     float L0, dh;
     bdry_cal_abl_len_dh(gd,D[n].ni1,
                            D[n].ni2,
                            D[n].nj1,
                            D[n].nj2,
                            D[n].nk1,
                            D[n].nk2,
                            2,
                            &L0, &dh);

     for (int k=D[n].nk1; k<=D[n].nk2; k++)
     {
        bdryexp->ablexp_Ez[k] = bdry_ablexp_cal_mask(k - D[n].nk1 + 1,
                                                     in_velocity[2][1], dt,
                                                     D[n].nk, dh);
     }
  }


  return ierr;
}

float
bdry_ablexp_cal_mask(int i, float vel, float dt, int num_lay, float dh)
{
  float len = num_lay * dh;

  int num_step  = (int) (len / vel / dt);

  float total_damp=0.0;
  for (int n=0; n<num_step; n++) {
     total_damp += powf((n*dt*vel)/len, 2.0);
  }

  float alpha = 0.6 / total_damp;

  float mask_val = expf( -alpha * powf((float)i/num_lay, 2.0 ) );

  return mask_val;
}

int
bdry_ablexp_apply(bdry_t bdry, gdinfo_t *gdinfo, float *w_end, int ncmp)
{
  float *Ex = bdry.ablexp_Ex;
  float *Ey = bdry.ablexp_Ey;
  float *Ez = bdry.ablexp_Ez;

  size_t siz_line   = gdinfo->siz_line;
  size_t siz_slice  = gdinfo->siz_slice;
  size_t siz_volume = gdinfo->siz_volume;


  bdry_block_t *D = bdry.bdry_blk;

  for (int n=0; n < CONST_NDIM_2; n++)
  {
    int ni = D[n].ni;
    int nj = D[n].nj;
    int nk = D[n].nk;

    int ni1 = D[n].ni1;
    int nj1 = D[n].nj1;
    int nk1 = D[n].nk1;

    if (D[n].enable == 1)
    {
      dim3 block(8,8,8);
      dim3 grid;
      grid.x = (ni + block.x - 1) / block.x;
      grid.y = (nj + block.y - 1) / block.y;
      grid.z = (nk + block.z - 1) / block.z;
      bdry_ablexp_apply_gpu<<<grid, block>>> (
                            Ex, Ey, Ez, 
                            w_end, ncmp, 
                            ni1, nj1, nk1, ni, nj, nk,  
                            siz_line, siz_slice, siz_volume);
    }
  }

  return 0;
}

__global__ void
bdry_ablexp_apply_gpu(float *Ex, float *Ey, float *Ez, 
                      float *w_end, int ncmp, 
                      int ni1, int nj1, int nk1, int ni, int nj, int nk,
                      size_t siz_line, size_t siz_slice, size_t siz_volume)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  float mask;
  size_t iptr;
  if(ix<ni && iy<nj && iz<nk)
  {
    iptr = (iz+nk1) * siz_slice + (iy+nj1) * siz_line + (ix+ni1);
    mask = (Ex[ix+ni1]<Ey[iy+nj1]) ? Ex[ix+ni1] : Ey[iy+nj1];
    if (mask > Ez[iz+nk1]) mask = Ez[iz+nk1];
    // unroll for accelate 
    // ncmp=9
    w_end[iptr + 0 * siz_volume] *= mask;
    w_end[iptr + 1 * siz_volume] *= mask;
    w_end[iptr + 2 * siz_volume] *= mask;
    w_end[iptr + 3 * siz_volume] *= mask;
    w_end[iptr + 4 * siz_volume] *= mask;
    w_end[iptr + 5 * siz_volume] *= mask;
    w_end[iptr + 6 * siz_volume] *= mask;
    w_end[iptr + 7 * siz_volume] *= mask;
    w_end[iptr + 8 * siz_volume] *= mask;
  }

  return;
}
