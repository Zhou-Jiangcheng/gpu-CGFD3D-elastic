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
#include "blk_t.h"
#include "sv_curv_col_el_gpu.h"
#include "sv_curv_col_ac_iso_gpu.h"
#include "sv_curv_col_el_iso_gpu.h"
#include "sv_curv_col_el_vti_gpu.h"
#include "sv_curv_col_el_aniso_gpu.h"
#include "sv_curv_col_vis_iso_gpu.h"
#include "alloc.h"
#include "cuda_common.h"

/*******************************************************************************
 * one simulation over all time steps, could be used in imaging or inversion
 *  simple MPI exchange without computing-communication overlapping
 ******************************************************************************/

int
drv_rk_curv_col_allstep(
  fd_t        *fd,
  gd_t        *gd,
  gd_metric_t *metric,
  md_t      *md,
  src_t     *src,
  bdryfree_t *bdryfree,
  bdrypml_t  *bdrypml,
  bdryexp_t  *bdryexp,
  wav_t     *wav,
  mympi_t   *mympi,
  iorecv_t  *iorecv,
  ioline_t  *ioline,
  ioslice_t *ioslice,
  iosnap_t  *iosnap,
  // time
  float dt, int nt_total, float t0,
  char *output_fname_part,
  char *output_dir,
  int qc_check_nan_num_of_step,
  const int output_all)
{
  // retrieve from struct
  int num_rk_stages = fd->num_rk_stages;
  float *rk_a = fd->rk_a;
  float *rk_b = fd->rk_b;

  int num_of_pairs     = fd->num_of_pairs;
  int fdz_max_len      = fd->fdz_max_len;
  int num_of_fdz_op    = fd->num_of_fdz_op;
  
  //gdinfo
  int ni = gd->ni;
  int nj = gd->nj;
  int nk = gd->nk;

  // mpi
  int myid = mympi->myid;
  int *topoid = mympi->topoid;
  MPI_Comm comm = mympi->comm;
  int *neighid_d = init_neighid_device(mympi->neighid);
  // local allocated array
  char ou_file[CONST_MAX_STRLEN];

  // calculate conversion matrix for free surface
  // use CPU calculate, then copy to gpu
  if (bdryfree->is_sides_free[CONST_NDIM-1][1] == 1)
  {
    if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO)
    {
      sv_curv_col_el_iso_dvh2dvz(gd,metric,md,bdryfree);
    } else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI) {
      sv_curv_col_el_vti_dvh2dvz(gd,metric,md,bdryfree);
    } else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO) {
      sv_curv_col_el_aniso_dvh2dvz(gd,metric,md,bdryfree);
    } else if (md->medium_type == CONST_MEDIUM_VISCOELASTIC_ISO) {
      if(md->visco_type == CONST_VISCO_GMB)
      {
        sv_curv_col_vis_iso_dvh2dvz(gd,metric,md,bdryfree,fd->fdc_len,fd->fdc_indx,fd->fdc_coef);
      }
    } else if (md->medium_type == CONST_MEDIUM_ACOUSTIC_ISO) {
      // not need
    } else {
      fprintf(stderr,"ERROR: conversion matrix for medium_type=%d is not implemented\n",
                    md->medium_type);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  gd_t   gd_d;
  md_t   md_d;
  src_t  src_d;
  wav_t  wav_d;
  gd_metric_t metric_d;
  fd_wav_t fd_wav_d;
  bdryfree_t  bdryfree_d;
  bdrypml_t   bdrypml_d;
  bdryexp_t   bdryexp_d;

  // init device struct, and copy data from host to device
  init_gdinfo_device(gd, &gd_d);
  init_md_device(md, &md_d);
  init_fd_device(fd, &fd_wav_d, gd);
  init_src_device(src, &src_d);
  init_metric_device(metric, &metric_d);
  init_wave_device(wav, &wav_d);
  init_bdryfree_device(gd, bdryfree, &bdryfree_d);
  init_bdrypml_device(gd, bdrypml, &bdrypml_d);
  init_bdryexp_device(gd, bdryexp, &bdryexp_d);

  //---------------------------------------
  // get device wavefield 
  float *w_buff = wav->v5d; // size number is V->siz_icmp * (V->ncmp+6)
  // GPU local pointer
  float *w_cur_d;
  float *w_pre_d;
  float *w_rhs_d;
  float *w_end_d;
  float *w_tmp_d;

  // get wavefield
  w_pre_d = wav_d.v5d + wav_d.siz_ilevel * 0; // previous level at n
  w_tmp_d = wav_d.v5d + wav_d.siz_ilevel * 1; // intermidate value
  w_rhs_d = wav_d.v5d + wav_d.siz_ilevel * 2; // for rhs
  w_end_d = wav_d.v5d + wav_d.siz_ilevel * 3; // end level at n+1
  int   ipair, istage;
  float t_cur;
  float t_end; // time after this loop for nc output
  // for mpi message
  int   ipair_mpi, istage_mpi;
  // create slice nc output files
  if (myid==0) fprintf(stdout,"prepare slice nc output ...\n"); 
  ioslice_nc_t ioslice_nc;
  io_slice_nc_create(ioslice, wav_d.ncmp, md_d.visco_type, wav_d.cmp_name,
                     gd_d.ni, gd_d.nj, gd_d.nk, topoid,
                     &ioslice_nc);
  // create snapshot nc output files
  if (myid==0) fprintf(stdout,"prepare snap nc output ...\n"); 
  iosnap_nc_t  iosnap_nc;
  if (md->medium_type == CONST_MEDIUM_ACOUSTIC_ISO) {
    io_snap_nc_create_ac(iosnap, &iosnap_nc, topoid);
  } else {
    io_snap_nc_create(iosnap, &iosnap_nc, topoid);
  }

  // only x/y mpi
  int num_of_r_reqs = 4;
  int num_of_s_reqs = 4;
  
  // set pml for rk
  if(bdrypml_d.is_enable_pml == 1)
  {
    for (int idim=0; idim<CONST_NDIM; idim++) {
      for (int iside=0; iside<2; iside++) {
        if (bdrypml_d.is_sides_pml[idim][iside]==1) {
          bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
          auxvar_d->pre = auxvar_d->var + auxvar_d->siz_ilevel * 0;
          auxvar_d->tmp = auxvar_d->var + auxvar_d->siz_ilevel * 1;
          auxvar_d->rhs = auxvar_d->var + auxvar_d->siz_ilevel * 2;
          auxvar_d->end = auxvar_d->var + auxvar_d->siz_ilevel * 3;
        }
      }
    }
  }

  // alloc free surface PGV, PGA and PGD
  float *PG_d = NULL;
  float *PG   = NULL;
  // Dis_accu is Displacemen accumulation, be uesd for PGD calculaton.
  float *Dis_accu_d   = NULL;
  if (bdryfree->is_sides_free[CONST_NDIM-1][1] == 1)
  {
    PG_d = init_PGVAD_device(gd);
    Dis_accu_d = init_Dis_accu_device(gd);
    PG = (float *) fdlib_mem_calloc_1d_float(CONST_NDIM_5*gd->ny*gd->nx,0.0,"PGV,A,D malloc");
  }

  //--------------------------------------------------------
  // time loop
  //--------------------------------------------------------

  if (myid==0) fprintf(stdout,"start time loop ...\n"); 

  //---------
  for (int it=0; it<nt_total; it++)
  {
    t_cur  = it * dt + t0;
    t_end = t_cur +dt;

    // output t=0 wavefiled
    //-- recv by interp
    io_recv_keep(iorecv, w_pre_d, w_buff, it, wav->ncmp, wav->siz_icmp);

    //-- line values
    io_line_keep(ioline, w_pre_d, w_buff, it, wav->ncmp, wav->siz_icmp);

    // write slice, use w_rhs as buff
    io_slice_nc_put(ioslice,&ioslice_nc,gd,w_pre_d,w_buff,it,t_cur);

    // snapshot
    if (md->medium_type == CONST_MEDIUM_ACOUSTIC_ISO) {
      io_snap_nc_put_ac(iosnap, &iosnap_nc, gd, md, wav, 
                     w_pre_d, w_buff, nt_total, it, t_cur);
    } else {
      io_snap_nc_put(iosnap, &iosnap_nc, gd, md, wav, 
                     w_pre_d, w_buff, nt_total, it, t_cur);
    }


    if (myid==0 && it%10== 0) fprintf(stdout,"-> it=%d, t=%f\n", it, t_cur);

    // mod to get ipair
    ipair = it % num_of_pairs;

    // loop RK stages for one step
    for (istage=0; istage<num_rk_stages; istage++)
    {

      // for mesg
      if (istage != num_rk_stages-1) {
        ipair_mpi = ipair;
        istage_mpi = istage + 1;
      } else {
        ipair_mpi = (it + 1) % num_of_pairs;
        istage_mpi = 0; 
      }

      // use pointer to avoid 1 copy for previous level value
      if (istage==0) {
        w_cur_d = w_pre_d;
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              bdrypml_d.auxvar[idim][iside].cur = bdrypml_d.auxvar[idim][iside].pre;
            }
          }
        }
      } else {
        w_cur_d = w_tmp_d;
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              bdrypml_d.auxvar[idim][iside].cur = bdrypml_d.auxvar[idim][iside].tmp;
            }
          }
        }
      }

      // set src_t time
      src_set_time(&src_d, it, istage);

      // compute rhs
      switch (md_d.medium_type)
      {
        case CONST_MEDIUM_ACOUSTIC_ISO : {

          sv_curv_col_ac_iso_onestage(
              w_cur_d,w_rhs_d,wav_d,fd_wav_d,
              gd_d, metric_d, md_d, bdrypml_d, bdryfree_d, src_d,
              fd->num_of_fdx_op, fd->pair_fdx_op[ipair][istage],
              fd->num_of_fdy_op, fd->pair_fdy_op[ipair][istage],
              fd->num_of_fdz_op, fd->pair_fdz_op[ipair][istage],
              fd->fdz_max_len,
              myid);
          break;
        }

        case CONST_MEDIUM_ELASTIC_ISO : {

          sv_curv_col_el_iso_onestage(
              w_cur_d,w_rhs_d,wav_d,fd_wav_d,
              gd_d, metric_d, md_d, bdrypml_d, bdryfree_d, src_d,
              fd->num_of_fdx_op, fd->pair_fdx_op[ipair][istage],
              fd->num_of_fdy_op, fd->pair_fdy_op[ipair][istage],
              fd->num_of_fdz_op, fd->pair_fdz_op[ipair][istage],
              fd->fdz_max_len,
              myid);
          break;
        }

        case CONST_MEDIUM_ELASTIC_VTI : {

          sv_curv_col_el_vti_onestage(
              w_cur_d,w_rhs_d,wav_d,fd_wav_d,
              gd_d, metric_d, md_d, bdrypml_d, bdryfree_d, src_d,
              fd->num_of_fdx_op, fd->pair_fdx_op[ipair][istage],
              fd->num_of_fdy_op, fd->pair_fdy_op[ipair][istage],
              fd->num_of_fdz_op, fd->pair_fdz_op[ipair][istage],
              fd->fdz_max_len,
              myid);
          break;
        }

        case CONST_MEDIUM_ELASTIC_ANISO : {

          sv_curv_col_el_aniso_onestage(
              w_cur_d,w_rhs_d,wav_d,fd_wav_d,
              gd_d, metric_d, md_d, bdrypml_d, bdryfree_d, src_d,
              fd->num_of_fdx_op, fd->pair_fdx_op[ipair][istage],
              fd->num_of_fdy_op, fd->pair_fdy_op[ipair][istage],
              fd->num_of_fdz_op, fd->pair_fdz_op[ipair][istage],
              fd->fdz_max_len,
              myid);
          break;
        }

        case CONST_MEDIUM_VISCOELASTIC_ISO : {

          if(md_d.visco_type == CONST_VISCO_GMB)
          {
            sv_curv_col_vis_iso_onestage(
                w_cur_d,w_rhs_d,wav_d,fd_wav_d,
                gd_d, metric_d, md_d, bdrypml_d, bdryfree_d, src_d,
                fd->num_of_fdx_op, fd->pair_fdx_op[ipair][istage],
                fd->num_of_fdy_op, fd->pair_fdy_op[ipair][istage],
                fd->num_of_fdz_op, fd->pair_fdz_op[ipair][istage],
                fd->fdz_max_len,
                myid);
          }
          break;
        }
      //  synchronize onestage device func.
      CUDACHECK(cudaDeviceSynchronize());
      }
      // recv mesg
      MPI_Startall(num_of_r_reqs, mympi->pair_r_reqs[ipair_mpi][istage_mpi]);

      // rk start
      if (istage==0)
      {
        float coef_a = rk_a[istage] * dt;
        float coef_b = rk_b[istage] * dt;

        // wavefield
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_a, w_tmp_d, w_pre_d, w_rhs_d);
        }

        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_tmp_d, fd, gd, mympi, wav_d.ncmp, ipair_mpi, istage_mpi, myid);

        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        
        // pml_tmp
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_a, auxvar_d->tmp, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
        // w_end
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_b, w_end_d, w_pre_d, w_rhs_d);
        }
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                            auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
      } else if (istage<num_rk_stages-1) {
        float coef_a = rk_a[istage] * dt;
        float coef_b = rk_b[istage] * dt;
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update <<<grid, block>>> (wav_d.siz_ilevel, coef_a, w_tmp_d, w_pre_d, w_rhs_d);
        }

        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_tmp_d, fd, gd, mympi, wav_d.ncmp, ipair_mpi, istage_mpi, myid);
        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        // pml_tmp
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_a, auxvar_d->tmp, auxvar_d->pre, auxvar_d->rhs);
              }
            }
          }
        }
        // w_end
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update_end <<<grid, block>>> (wav_d.siz_ilevel, coef_b, w_end_d, w_rhs_d);
        }
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update_end <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->rhs);
              }
            }
          }
        }
      } else { // last stage 
        float coef_b = rk_b[istage] * dt;

        // wavefield
        {
          dim3 block(256);
          dim3 grid;
          grid.x = (wav_d.siz_ilevel + block.x - 1) / block.x;
          wav_update_end <<<grid, block>>>(wav_d.siz_ilevel, coef_b, w_end_d, w_rhs_d);
        }

        
        // pack and isend
        blk_macdrp_pack_mesg_gpu(w_end_d, fd, gd, mympi, wav_d.ncmp, ipair_mpi, istage_mpi, myid);
        MPI_Startall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi]);
        // pml_end
        if(bdrypml_d.is_enable_pml == 1)
        {
          for (int idim=0; idim<CONST_NDIM; idim++) {
            for (int iside=0; iside<2; iside++) {
              if (bdrypml_d.is_sides_pml[idim][iside]==1) {
                bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
                dim3 block(256);
                dim3 grid;
                grid.x = (auxvar_d->siz_ilevel + block.x - 1) / block.x;
                wav_update_end <<<grid, block>>> (
                           auxvar_d->siz_ilevel, coef_b, auxvar_d->end, auxvar_d->rhs);
              }
            }
          }
        }
        if (md->medium_type == CONST_MEDIUM_VISCOELASTIC_ISO) 
        {
          if (bdryfree->is_sides_free[2][1] == 1) 
          {
            dim3 block(32,16);
            dim3 grid;
            grid.x = (ni+block.x-1)/block.x;
            grid.y = (nj+block.y-1)/block.y;
            sv_curv_col_vis_iso_free_gpu <<<grid, block>>> (
                w_end_d,wav_d,gd_d,metric_d,md_d,bdryfree_d,myid);
          }
        }
      }
      MPI_Waitall(num_of_s_reqs, mympi->pair_s_reqs[ipair_mpi][istage_mpi], MPI_STATUS_IGNORE);
      MPI_Waitall(num_of_r_reqs, mympi->pair_r_reqs[ipair_mpi][istage_mpi], MPI_STATUS_IGNORE);
 
      if (istage != num_rk_stages-1) 
      {
        blk_macdrp_unpack_mesg_gpu(w_tmp_d, fd, gd, mympi, wav_d.ncmp, ipair_mpi, istage_mpi, neighid_d);
      } else 
      {
        blk_macdrp_unpack_mesg_gpu(w_end_d, fd, gd, mympi, wav_d.ncmp, ipair_mpi, istage_mpi, neighid_d);
      }
    } // RK stages

    //--------------------------------------------
    // QC
    //--------------------------------------------
    if (qc_check_nan_num_of_step >0  && (it % qc_check_nan_num_of_step) == 0) {
      if (myid==0) fprintf(stdout,"-> check value nan\n");
        //wav_check_value(w_end);
    }

    //--------------------------------------------
     if (bdryexp_d.is_enable_ablexp == 1) {
       bdry_ablexp_apply(bdryexp_d, gd, w_end_d, wav->ncmp);
     }

    // save results
    //--------------------------------------------
    // calculate PGV, PGA and PGD for each surface at each stage
    if (bdryfree->is_sides_free[CONST_NDIM-1][1] == 1)
    {
        dim3 block(8,8);
        dim3 grid;
        grid.x = (ni + block.x - 1) / block.x;
        grid.y = (nj + block.y - 1) / block.y;
        PG_calcu_gpu<<<grid, block>>> (w_end_d, w_pre_d, gd_d, PG_d, Dis_accu_d, dt);
    }
    // swap w_pre and w_end, avoid copying
    w_cur_d = w_pre_d; w_pre_d = w_end_d; w_end_d = w_cur_d;

    if(bdrypml_d.is_enable_pml == 1)
    {
      for (int idim=0; idim<CONST_NDIM; idim++) {
        for (int iside=0; iside<2; iside++) {
          bdrypml_auxvar_t *auxvar_d = &(bdrypml_d.auxvar[idim][iside]);
          auxvar_d->cur = auxvar_d->pre;
          auxvar_d->pre = auxvar_d->end;
          auxvar_d->end = auxvar_d->cur;
        }
      }
    }
  } // time loop

  cudaMemcpy(PG,PG_d,sizeof(float)*CONST_NDIM_5*gd->ny*gd->nx,cudaMemcpyDeviceToHost);
  // finish all time loop calculate, cudafree device pointer
  CUDACHECK(cudaFree(PG_d));
  CUDACHECK(cudaFree(Dis_accu_d));
  CUDACHECK(cudaFree(neighid_d));
  dealloc_md_device(md_d);
  dealloc_fd_device(fd_wav_d);
  dealloc_metric_device(metric_d);
  dealloc_src_device(src_d);
  dealloc_wave_device(wav_d);
  dealloc_bdryfree_device(bdryfree_d);
  dealloc_bdrypml_device(bdrypml_d);
  dealloc_bdryexp_device(bdryexp_d);
  // postproc
  if (bdryfree->is_sides_free[CONST_NDIM-1][1] == 1)
  {
    PG_slice_output(PG,gd,output_dir,output_fname_part,topoid);
  }
  // close nc
  io_slice_nc_close(&ioslice_nc);
  io_snap_nc_close(&iosnap_nc);

  return 0;
}
