#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "alloc.h"
#include "cuda_common.h"


int init_gdinfo_device(gdinfo_t *gdinfo, gdinfo_t *gdinfo_d)
{
  memcpy(gdinfo_d,gdinfo,sizeof(gdinfo_t));
  return 0;
}

int init_gdcart_device(gd_t *gdcart, gd_t *gdcart_d)
{
  memcpy(gdcart_d,gdcart,sizeof(gd_t));
  return 0;
}

int init_gdcurv_device(gd_t *gdcurv, gd_t *gdcurv_d)
{
  size_t siz_icmp = gdcurv->siz_icmp;
  memcpy(gdcurv_d,gdcurv,sizeof(gd_t));
  gdcurv_d->x3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->y3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->z3d = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gdcurv_d->cell_xmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_xmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_ymin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_ymax = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_zmin = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  gdcurv_d->cell_zmax = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  gdcurv_d->tile_istart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gdcurv_d->tile_iend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NX);
  gdcurv_d->tile_jstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gdcurv_d->tile_jend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NY);
  gdcurv_d->tile_kstart = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);
  gdcurv_d->tile_kend   = (int *) cuda_malloc(sizeof(int)*GD_TILE_NZ);

  int size = GD_TILE_NX * GD_TILE_NY * GD_TILE_NZ;
  gdcurv_d->tile_xmin   = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_xmax   = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_ymin   = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_ymax   = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_zmin   = (float *) cuda_malloc(sizeof(float)*size);
  gdcurv_d->tile_zmax   = (float *) cuda_malloc(sizeof(float)*size);

  CUDACHECK(cudaMemcpy(gdcurv_d->x3d, gdcurv->x3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->y3d, gdcurv->y3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->z3d, gdcurv->z3d, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmin, gdcurv->cell_xmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_xmax, gdcurv->cell_xmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymin, gdcurv->cell_ymin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_ymax, gdcurv->cell_ymax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmin, gdcurv->cell_zmin, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->cell_zmax, gdcurv->cell_zmax, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->tile_istart, gdcurv->tile_istart, sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_iend,   gdcurv->tile_iend,   sizeof(int)*GD_TILE_NX, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_jstart, gdcurv->tile_jstart, sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_jend,   gdcurv->tile_jend,   sizeof(int)*GD_TILE_NY, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_kstart, gdcurv->tile_kstart, sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_kend,   gdcurv->tile_kend,   sizeof(int)*GD_TILE_NZ, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMemcpy(gdcurv_d->tile_xmin,   gdcurv->tile_xmin,   sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_xmax,   gdcurv->tile_xmax,   sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_ymin,   gdcurv->tile_ymin,   sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_ymax,   gdcurv->tile_ymax,   sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_zmin,   gdcurv->tile_zmin,   sizeof(float)*size, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(gdcurv_d->tile_zmax,   gdcurv->tile_zmax,   sizeof(float)*size, cudaMemcpyHostToDevice));

  return 0;
}
int init_md_device(md_t *md, md_t *md_d)
{
  size_t siz_icmp = md->siz_icmp;

  memcpy(md_d,md,sizeof(md_t));
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->lambda = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->mu     = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->lambda, md->lambda, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->mu,     md->mu,     sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }
  if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO)
  {
    md_d->rho    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c11    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c12    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c13    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c14    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c15    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c16    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c22    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c23    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c24    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c25    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c26    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c33    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c34    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c35    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c36    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c44    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c45    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c46    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c55    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c56    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    md_d->c66    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
    CUDACHECK(cudaMemcpy(md_d->rho,    md->rho,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c11,    md->c11,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c12,    md->c12,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c13,    md->c13,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c14,    md->c14,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c15,    md->c15,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c16,    md->c16,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c22,    md->c22,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c23,    md->c23,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c24,    md->c24,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c25,    md->c25,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c26,    md->c26,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c33,    md->c33,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c34,    md->c34,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c35,    md->c35,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c36,    md->c36,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c44,    md->c44,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c45,    md->c45,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c46,    md->c46,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c55,    md->c55,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c56,    md->c56,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(md_d->c66,    md->c66,    sizeof(float)*siz_icmp, cudaMemcpyHostToDevice));
  }

  return 0;
}

int init_fd_device(fd_t *fd, fd_wav_t *fd_wav_d)
{
  int max_len = fd->fdz_max_len; //=5 
  int max_lay = fd->num_of_fdz_op;
  fd_wav_d->fdz_len_d     = (int *) cuda_malloc(sizeof(int)*max_lay);
  fd_wav_d->fdx_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);
  fd_wav_d->fdy_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);
  fd_wav_d->fdz_coef_d    = (float *) cuda_malloc(sizeof(float)*max_len);
  fd_wav_d->fdz_coef_all_d    = (float *) cuda_malloc(sizeof(float)*max_len*max_lay);

  fd_wav_d->fdx_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);
  fd_wav_d->fdy_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);
  fd_wav_d->fdz_indx_d    = (int *) cuda_malloc(sizeof(int)*max_len);
  fd_wav_d->fdz_indx_all_d    = (int *) cuda_malloc(sizeof(int)*max_len*max_lay);

  fd_wav_d->fdx_shift_d    = (size_t *) cuda_malloc(sizeof(size_t)*max_len);
  fd_wav_d->fdy_shift_d    = (size_t *) cuda_malloc(sizeof(size_t)*max_len);
  fd_wav_d->fdz_shift_d    = (size_t *) cuda_malloc(sizeof(size_t)*max_len);
  fd_wav_d->fdz_shift_all_d    = (size_t *) cuda_malloc(sizeof(size_t)*max_len*max_lay);
  return 0;
}

int init_metric_device(gdcurv_metric_t *metric, gdcurv_metric_t *metric_d)
{
  size_t siz_icmp = metric->siz_icmp;

  memcpy(metric_d,metric,sizeof(gdcurv_metric_t));
  metric_d->jac     = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_x    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_y    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->xi_z    = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_x   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_y   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->eta_z   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_x   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_y   = (float *) cuda_malloc(sizeof(float)*siz_icmp);
  metric_d->zeta_z   = (float *) cuda_malloc(sizeof(float)*siz_icmp);

  CUDACHECK( cudaMemcpy(metric_d->jac,   metric->jac,   sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_x,  metric->xi_x,  sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_y,  metric->xi_y,  sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->xi_z,  metric->xi_z,  sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_x, metric->eta_x, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_y, metric->eta_y, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->eta_z, metric->eta_z, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_x, metric->zeta_x, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_y, metric->zeta_y, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaMemcpy(metric_d->zeta_z, metric->zeta_z, sizeof(float)*siz_icmp, cudaMemcpyHostToDevice) );
  return 0;
}

int init_src_device(src_t *src, src_t *src_d)
{
  int total_number = src->total_number;
  int max_ext      = src->max_ext;
  size_t temp_all     = (src->total_number) * (src->max_nt) * (src->max_stage);

  memcpy(src_d,src,sizeof(src_t));
  if(src->force_actived == 1) {
    src_d->Fx  = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Fy  = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Fz  = (float *) cuda_malloc(sizeof(float)*temp_all);
    CUDACHECK( cudaMemcpy(src_d->Fx,  src->Fx,  sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Fy,  src->Fy,  sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Fz,  src->Fz,  sizeof(float)*temp_all, cudaMemcpyHostToDevice));
  }
  if(src->moment_actived == 1) {
    src_d->Mxx = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Myy = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Mzz = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Mxz = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Myz = (float *) cuda_malloc(sizeof(float)*temp_all);
    src_d->Mxy = (float *) cuda_malloc(sizeof(float)*temp_all);
    CUDACHECK( cudaMemcpy(src_d->Mxx, src->Mxx, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Myy, src->Myy, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Mzz, src->Mzz, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Mxz, src->Mxz, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Myz, src->Myz, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->Mxy, src->Mxy, sizeof(float)*temp_all, cudaMemcpyHostToDevice));
  }
  if(total_number>0)
  {
    src_d->ext_num  = (int *) cuda_malloc(sizeof(int)*total_number);
    src_d->it_begin = (int *) cuda_malloc(sizeof(int)*total_number);
    src_d->it_end   = (int *) cuda_malloc(sizeof(int)*total_number);
    src_d->ext_indx = (int *) cuda_malloc(sizeof(int)*total_number*max_ext);
    src_d->ext_coef = (float *) cuda_malloc(sizeof(float)*total_number*max_ext);

    CUDACHECK( cudaMemcpy(src_d->ext_num,  src->ext_num,  sizeof(int)*total_number, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->it_begin, src->it_begin, sizeof(int)*total_number, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->it_end,   src->it_end,   sizeof(int)*total_number, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->ext_indx, src->ext_indx, sizeof(int)*total_number*max_ext, cudaMemcpyHostToDevice));
    CUDACHECK( cudaMemcpy(src_d->ext_coef, src->ext_coef, sizeof(float)*total_number*max_ext, cudaMemcpyHostToDevice));
  }
  return 0;
}

int init_bdry_device(gdinfo_t *gdinfo, bdry_t *bdry, bdry_t *bdry_d)
{
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
  int nz = gdinfo->nz;

  memcpy(bdry_d,bdry,sizeof(bdry_t));
  // copy bdryfree
  if (bdry_d->is_sides_free[CONST_NDIM-1][1] == 1)
  {
    bdry_d->matVx2Vz2   = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);
    bdry_d->matVy2Vz2   = (float *) cuda_malloc(sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM);

    CUDACHECK(cudaMemcpy(bdry_d->matVx2Vz2, bdry->matVx2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdry_d->matVy2Vz2, bdry->matVy2Vz2, sizeof(float)*nx*ny*CONST_NDIM*CONST_NDIM, cudaMemcpyHostToDevice));
  }

  // copy bdrypml
  if (bdry_d->is_enable_pml == 1)
  {
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        if(bdry_d->is_sides_pml[idim][iside] == 1){
          int npoints = bdry_d->num_of_layers[idim][iside] + 1;
          bdry_d->A[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          bdry_d->B[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          bdry_d->D[idim][iside]   = (float *) cuda_malloc(npoints * sizeof(float));
          CUDACHECK(cudaMemcpy(bdry_d->A[idim][iside],bdry->A[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(bdry_d->B[idim][iside],bdry->B[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          CUDACHECK(cudaMemcpy(bdry_d->D[idim][iside],bdry->D[idim][iside],npoints*sizeof(float),cudaMemcpyHostToDevice));
          } else {
          bdry_d->A[idim][iside] = NULL;
          bdry_d->B[idim][iside] = NULL;
          bdry_d->D[idim][iside] = NULL;
        }
      }
    }

    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        bdrypml_auxvar_t *auxvar_d = &(bdry_d->auxvar[idim][iside]);
        if(auxvar_d->siz_icmp > 0){
          auxvar_d->var = (float *) cuda_malloc(sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel); 
          CUDACHECK(cudaMemset(auxvar_d->var,0,sizeof(float)*auxvar_d->siz_ilevel*auxvar_d->nlevel));
        } else {
        auxvar_d->var = NULL;
        }
      }
    }
  }
  // copy bdryexp
  if (bdry_d->is_enable_ablexp == 1)
  {
    bdry_d->ablexp_Ex = (float *) cuda_malloc(nx * sizeof(float));
    bdry_d->ablexp_Ey = (float *) cuda_malloc(ny * sizeof(float));
    bdry_d->ablexp_Ez = (float *) cuda_malloc(nz * sizeof(float));
    CUDACHECK(cudaMemcpy(bdry_d->ablexp_Ex,bdry->ablexp_Ex,nx*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdry_d->ablexp_Ey,bdry->ablexp_Ey,ny*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(bdry_d->ablexp_Ez,bdry->ablexp_Ez,nz*sizeof(float),cudaMemcpyHostToDevice));
  }

  return 0;
}


int init_wave_device(wav_t *wav, wav_t *wav_d)
{
  size_t siz_ilevel = wav->siz_ilevel;
  int nlevel = wav->nlevel;
  memcpy(wav_d,wav,sizeof(wav_t));
  wav_d->v5d   = (float *) cuda_malloc(sizeof(float)*siz_ilevel*nlevel);
  CUDACHECK(cudaMemset(wav_d->v5d,0,sizeof(float)*siz_ilevel*nlevel));

  return 0;
}

float *init_PGVAD_device(gdinfo_t *gdinfo)
{
  float *PG_d;
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
  PG_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM_5*nx*ny);
  CUDACHECK(cudaMemset(PG_d,0,sizeof(float)*CONST_NDIM_5*nx*ny));

  return PG_d;
}

float *init_Dis_accu_device(gdinfo_t *gdinfo)
{
  float *Dis_accu_d;
  int nx = gdinfo->nx;
  int ny = gdinfo->ny;
  Dis_accu_d = (float *) cuda_malloc(sizeof(float)*CONST_NDIM*nx*ny);
  CUDACHECK(cudaMemset(Dis_accu_d,0,sizeof(float)*CONST_NDIM*nx*ny));

  return Dis_accu_d;
}

int *init_neighid_device(int *neighid)
{
  int *neighid_d; 
  neighid_d = (int *) cuda_malloc(sizeof(int)*CONST_NDIM_2);
  CUDACHECK(cudaMemcpy(neighid_d,neighid,sizeof(int)*CONST_NDIM_2,cudaMemcpyHostToDevice));

  return neighid_d;
}

int dealloc_gdcurv_device(gd_t gdcurv_d)
{
  CUDACHECK(cudaFree(gdcurv_d.x3d)); 
  CUDACHECK(cudaFree(gdcurv_d.y3d)); 
  CUDACHECK(cudaFree(gdcurv_d.z3d)); 

  CUDACHECK(cudaFree(gdcurv_d.cell_xmin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_xmax)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_ymin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_ymax)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_zmin)); 
  CUDACHECK(cudaFree(gdcurv_d.cell_zmax)); 
  return 0;
}

int dealloc_md_device(md_t md_d)
{
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_ISO)
  {
    CUDACHECK(cudaFree(md_d.rho   )); 
    CUDACHECK(cudaFree(md_d.lambda)); 
    CUDACHECK(cudaFree(md_d.mu    )); 
  }
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_VTI)
  {
    CUDACHECK(cudaFree(md_d.rho)); 
    CUDACHECK(cudaFree(md_d.c11)); 
    CUDACHECK(cudaFree(md_d.c33)); 
    CUDACHECK(cudaFree(md_d.c55)); 
    CUDACHECK(cudaFree(md_d.c66)); 
    CUDACHECK(cudaFree(md_d.c13)); 
  }
  if (md_d.medium_type == CONST_MEDIUM_ELASTIC_ANISO)
  {
    CUDACHECK(cudaFree(md_d.rho)); 
    CUDACHECK(cudaFree(md_d.c11)); 
    CUDACHECK(cudaFree(md_d.c12)); 
    CUDACHECK(cudaFree(md_d.c13)); 
    CUDACHECK(cudaFree(md_d.c14)); 
    CUDACHECK(cudaFree(md_d.c15)); 
    CUDACHECK(cudaFree(md_d.c16)); 
    CUDACHECK(cudaFree(md_d.c22)); 
    CUDACHECK(cudaFree(md_d.c23)); 
    CUDACHECK(cudaFree(md_d.c24)); 
    CUDACHECK(cudaFree(md_d.c25)); 
    CUDACHECK(cudaFree(md_d.c26)); 
    CUDACHECK(cudaFree(md_d.c33)); 
    CUDACHECK(cudaFree(md_d.c34)); 
    CUDACHECK(cudaFree(md_d.c35)); 
    CUDACHECK(cudaFree(md_d.c36)); 
    CUDACHECK(cudaFree(md_d.c44)); 
    CUDACHECK(cudaFree(md_d.c45)); 
    CUDACHECK(cudaFree(md_d.c46)); 
    CUDACHECK(cudaFree(md_d.c55)); 
    CUDACHECK(cudaFree(md_d.c56)); 
    CUDACHECK(cudaFree(md_d.c66)); 
  }

  return 0;
}

int dealloc_fd_device(fd_wav_t fd_wav_d)
{
  CUDACHECK(cudaFree(fd_wav_d.fdx_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_coef_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_coef_all_d));

  CUDACHECK(cudaFree(fd_wav_d.fdx_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_indx_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_indx_all_d));

  CUDACHECK(cudaFree(fd_wav_d.fdx_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdy_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_shift_d));
  CUDACHECK(cudaFree(fd_wav_d.fdz_shift_all_d));

  return 0;
}
int dealloc_metric_device(gdcurv_metric_t metric_d)
{
  CUDACHECK(cudaFree(metric_d.jac   )); 
  CUDACHECK(cudaFree(metric_d.xi_x  )); 
  CUDACHECK(cudaFree(metric_d.xi_y  )); 
  CUDACHECK(cudaFree(metric_d.xi_z  )); 
  CUDACHECK(cudaFree(metric_d.eta_x )); 
  CUDACHECK(cudaFree(metric_d.eta_y )); 
  CUDACHECK(cudaFree(metric_d.eta_z )); 
  CUDACHECK(cudaFree(metric_d.zeta_x)); 
  CUDACHECK(cudaFree(metric_d.zeta_y)); 
  CUDACHECK(cudaFree(metric_d.zeta_z)); 
  return 0;
}

int dealloc_src_device(src_t src_d)
{
  if(src_d.force_actived == 1)
  {
    CUDACHECK(cudaFree(src_d.Fx)); 
    CUDACHECK(cudaFree(src_d.Fy)); 
    CUDACHECK(cudaFree(src_d.Fz)); 
  }
  if(src_d.moment_actived == 1)
  {
    CUDACHECK(cudaFree(src_d.Mxx)); 
    CUDACHECK(cudaFree(src_d.Myy)); 
    CUDACHECK(cudaFree(src_d.Mzz)); 
    CUDACHECK(cudaFree(src_d.Mxz)); 
    CUDACHECK(cudaFree(src_d.Myz)); 
    CUDACHECK(cudaFree(src_d.Mxy)); 
  }
  if(src_d.total_number > 0)
  {
    CUDACHECK(cudaFree(src_d.ext_num )); 
    CUDACHECK(cudaFree(src_d.ext_indx)); 
    CUDACHECK(cudaFree(src_d.ext_coef)); 
    CUDACHECK(cudaFree(src_d.it_begin)); 
    CUDACHECK(cudaFree(src_d.it_end  )); 
  }
  return 0;
}

int dealloc_bdry_device(bdry_t bdry_d)
{
  if (bdry_d.is_sides_free[CONST_NDIM-1][1] == 1)
  {
    CUDACHECK(cudaFree(bdry_d.matVx2Vz2)); 
    CUDACHECK(cudaFree(bdry_d.matVy2Vz2)); 
  }
  if (bdry_d.is_enable_pml == 1)
  {
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        if(bdry_d.is_sides_pml[idim][iside] == 1){
          CUDACHECK(cudaFree(bdry_d.A[idim][iside])); 
          CUDACHECK(cudaFree(bdry_d.B[idim][iside])); 
          CUDACHECK(cudaFree(bdry_d.D[idim][iside])); 
        }
      }
    }  
    for(int idim=0; idim<CONST_NDIM; idim++){
      for(int iside=0; iside<2; iside++){
        bdrypml_auxvar_t *auxvar_d = &(bdry_d.auxvar[idim][iside]);
        if(auxvar_d->siz_icmp > 0){
          CUDACHECK(cudaFree(auxvar_d->var)); 
        }
      }
    }  
  }
  if (bdry_d.is_enable_ablexp == 1)
  {
    CUDACHECK(cudaFree(bdry_d.ablexp_Ex)); 
    CUDACHECK(cudaFree(bdry_d.ablexp_Ey));
    CUDACHECK(cudaFree(bdry_d.ablexp_Ez));
  }
  return 0;
}

int dealloc_wave_device(wav_t wav_d)
{
  CUDACHECK(cudaFree(wav_d.v5d)); 
  return 0;
}
