/*********************************************************************
 * setup fd operators
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fdlib_mem.h"
#include "fdlib_math.h"
#include "blk_t.h"
#include "cuda_common.h"

int
blk_init(blk_t *blk,
         const int myid, const int verbose)
{
  int ierr = 0;

  // alloc struct vars
  blk->fd            = (fd_t *)malloc(sizeof(fd_t));
  blk->mympi         = (mympi_t *)malloc(sizeof(mympi_t));
  blk->gdinfo        = (gdinfo_t *)malloc(sizeof(gdinfo_t));
  blk->gd            = (gd_t        *)malloc(sizeof(gd_t     ));
  blk->gdcurv_metric = (gdcurv_metric_t *)malloc(sizeof(gdcurv_metric_t));
  blk->md            = (md_t      *)malloc(sizeof(md_t     ));
  blk->wav           = (wav_t      *)malloc(sizeof(wav_t     ));
  blk->src           = (src_t      *)malloc(sizeof(src_t     ));
  blk->bdryfree      = (bdryfree_t *)malloc(sizeof(bdryfree_t ));
  blk->bdrypml       = (bdrypml_t  *)malloc(sizeof(bdrypml_t ));
  blk->bdryexp       = (bdryexp_t  *)malloc(sizeof(bdryexp_t ));
  blk->iorecv        = (iorecv_t   *)malloc(sizeof(iorecv_t ));
  blk->ioline        = (ioline_t   *)malloc(sizeof(ioline_t ));
  blk->ioslice       = (ioslice_t  *)malloc(sizeof(ioslice_t ));
  blk->iosnap        = (iosnap_t   *)malloc(sizeof(iosnap_t ));

  sprintf(blk->name, "%s", "single");

  return ierr;
}

// set str
int
blk_set_output(blk_t *blk,
               mympi_t *mympi,
               char *output_dir,
               char *grid_export_dir,
               char *media_export_dir,
               const int verbose)
{
  // set name
  //sprintf(blk->name, "%s", name);

  // output name
  sprintf(blk->output_fname_part,"px%d_py%d", mympi->topoid[0],mympi->topoid[1]);

  // output
  sprintf(blk->output_dir, "%s", output_dir);
  sprintf(blk->grid_export_dir, "%s", grid_export_dir);
  sprintf(blk->media_export_dir, "%s", media_export_dir);

  return 0;
}

int
blk_print(blk_t *blk)
{    
  int ierr = 0;

  fprintf(stdout, "\n-------------------------------------------------------\n");
  fprintf(stdout, "print blk %s:\n", blk->name);
  fprintf(stdout, "-------------------------------------------------------\n\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> ESTIMATE MEMORY INFO.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "total memory size Byte: %20.5f  B\n", PSV->total_memory_size_Byte);
  //fprintf(stdout, "total memory size KB  : %20.5f KB\n", PSV->total_memory_size_KB  );
  //fprintf(stdout, "total memory size MB  : %20.5f MB\n", PSV->total_memory_size_MB  );
  //fprintf(stdout, "total memory size GB  : %20.5f GB\n", PSV->total_memory_size_GB  );
  //fprintf(stdout, "\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> FOLDER AND FILE INFO.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "   OutFolderName: %s\n", OutFolderName);
  //fprintf(stdout, "       EventName: %s\n", OutPrefix);
  //fprintf(stdout, "     LogFilename: %s\n", LogFilename);
  //fprintf(stdout, " StationFilename: %s\n", StationFilename);
  //fprintf(stdout, "  SourceFilename: %s\n", SourceFilename);
  //fprintf(stdout, "   MediaFilename: %s\n", MediaFilename);
  //fprintf(stdout, "\n");

  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> media info.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //if (blk->media_type == MEDIA_TYPE_LAYER)
  //{
  //    strcpy(str, "layer");
  //}
  //else if (blk->media_type == MEDIA_TYPE_GRID)
  //{
  //    strcpy(str, "grid");
  //}
  //fprintf(stdout, " media_type = %s\n", str);
  //if(blk->media_type == MEDIA_TYPE_GRID)
  //{
  //    fprintf(stdout, "\n --> the media filename is:\n");
  //    fprintf(stdout, " velp_file  = %s\n", blk->fnm_velp);
  //    fprintf(stdout, " vels_file  = %s\n", blk->fnm_vels);
  //    fprintf(stdout, " rho_file   = %s\n", blk->fnm_rho);
  //}
  //fprintf(stdout, "\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> source info.\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, " number_of_force  = %d\n", blk->number_of_force);
  //if(blk->number_of_force > 0)
  //{
  //    fprintf(stdout, " force_source           x           z     x_shift     z_shift           i           k:\n");
  //    for(n=0; n<blk->number_of_force; n++)
  //    {
  //        indx = 2*n;
  //        fprintf(stdout, "         %04d  %10.4e  %10.4e  %10.4e  %10.4e  %10d  %10d\n", n+1, 
  //                blk->force_coord[indx], blk->force_coord[indx+1],
  //                blk->force_shift[indx], blk->force_shift[indx+1],
  //                blk->force_indx [indx], blk->force_indx [indx+1]);
  //    }
  //    fprintf(stdout, "\n");
  //}

  //fprintf(stdout, "\n");
  //fprintf(stdout, " number_of_moment = %d\n", blk->number_of_moment);
  //if(blk->number_of_moment > 0)
  //{
  //    fprintf(stdout, " moment_source          x           z     x_shift     z_shift           i           k:\n");
  //    for(n=0; n<blk->number_of_moment; n++)
  //    {
  //        indx = 2*n;
  //        fprintf(stdout, "         %04d  %10.4e  %10.4e  %10.4e  %10.4e  %10d  %10d\n", n+1, 
  //                blk->moment_coord[indx], blk->moment_coord[indx+1],
  //                blk->moment_shift[indx], blk->moment_shift[indx+1],
  //                blk->moment_indx [indx], blk->moment_indx [indx+1]);
  //    }
  //    fprintf(stdout, "\n");
  //}

  //fprintf(stdout, "-------------------------------------------------------\n");
  //fprintf(stdout, "--> boundary layer information:\n");
  //fprintf(stdout, "-------------------------------------------------------\n");
  //ierr = boundary_id2type(type1, blk->boundary_type[0], errorMsg);
  //ierr = boundary_id2type(type2, blk->boundary_type[1], errorMsg);
  //ierr = boundary_id2type(type3, blk->boundary_type[2], errorMsg);
  //ierr = boundary_id2type(type4, blk->boundary_type[3], errorMsg);
  //fprintf(stdout, " boundary_type         = %10s%10s%10s%10s\n", 
  //        type1, type2, type3, type4);
  //fprintf(stdout, " boundary_layer_number = %10d%10d%10d%10d\n", 
  //        blk->boundary_layer_number[0], blk->boundary_layer_number[1], 
  //        blk->boundary_layer_number[2], blk->boundary_layer_number[3]);
  //fprintf(stdout, "\n");
  //fprintf(stdout, " absorb_velocity       = %10.2f%10.2f%10.2f%10.2f\n", 
  //        blk->absorb_velocity[0], blk->absorb_velocity[1], blk->absorb_velocity[2], 
  //        blk->absorb_velocity[3]);
  //fprintf(stdout, "\n");
  //fprintf(stdout, " CFS_alpha_max         = %10.2f%10.2f%10.2f%10.2f\n", 
  //        blk->CFS_alpha_max[0], blk->CFS_alpha_max[1], blk->CFS_alpha_max[2], 
  //        blk->CFS_alpha_max[3]);
  //fprintf(stdout, " CFS_beta_max          = %10.2f%10.2f%10.2f%10.2f\n", 
  //        blk->CFS_beta_max[0], blk->CFS_beta_max[1], blk->CFS_beta_max[2], 
  //        blk->CFS_beta_max[3]);
  
  return ierr;
}

/*********************************************************************
 * mpi message for macdrp scheme with rk
 *********************************************************************/

void
blk_macdrp_mesg_init(mympi_t *mympi,
                fd_t *fd,
                int ni,
                int nj,
                int nk,
                int num_of_vars)
{
  // alloc
  mympi->pair_siz_sbuff_x1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_x2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_y1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_sbuff_y2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_x1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_x2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_y1 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_siz_rbuff_y2 = (size_t **)malloc(fd->num_of_pairs * sizeof(size_t *));
  mympi->pair_s_reqs       = (MPI_Request ***)malloc(fd->num_of_pairs * sizeof(MPI_Request **));
  mympi->pair_r_reqs       = (MPI_Request ***)malloc(fd->num_of_pairs * sizeof(MPI_Request **));
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    mympi->pair_siz_sbuff_x1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_x2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_y1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_sbuff_y2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_x1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_x2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_y1[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_siz_rbuff_y2[ipair] = (size_t *)malloc(fd->num_rk_stages * sizeof(size_t));
    mympi->pair_s_reqs[ipair] = (MPI_Request **)malloc(fd->num_rk_stages * sizeof(MPI_Request *));
    mympi->pair_r_reqs[ipair] = (MPI_Request **)malloc(fd->num_rk_stages * sizeof(MPI_Request *));

    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      mympi->pair_s_reqs[ipair][istage] = (MPI_Request *)malloc(4 * sizeof(MPI_Request));
      mympi->pair_r_reqs[ipair][istage] = (MPI_Request *)malloc(4 * sizeof(MPI_Request));
    }
  }

  // mpi mesg
  mympi->siz_sbuff = 0;
  mympi->siz_rbuff = 0;
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      fd_op_t *fdx_op = fd->pair_fdx_op[ipair][istage]+fd->num_of_fdx_op-1;
      fd_op_t *fdy_op = fd->pair_fdy_op[ipair][istage]+fd->num_of_fdy_op-1;

      // to x1, depends on right_len of x1 proc
      mympi->pair_siz_sbuff_x1[ipair][istage] = (nj * nk * fdx_op->right_len) * num_of_vars;
      // to x2, depends on left_len of x2 proc
      mympi->pair_siz_sbuff_x2[ipair][istage] = (nj * nk * fdx_op->left_len ) * num_of_vars;

      mympi->pair_siz_sbuff_y1[ipair][istage] = (ni * nk * fdy_op->right_len) * num_of_vars;
      mympi->pair_siz_sbuff_y2[ipair][istage] = (ni * nk * fdy_op->left_len ) * num_of_vars;

      // from x1, depends on left_len of cur proc
      mympi->pair_siz_rbuff_x1[ipair][istage] = (nj * nk * fdx_op->left_len ) * num_of_vars;
      // from x2, depends on right_len of cur proc
      mympi->pair_siz_rbuff_x2[ipair][istage] = (nj * nk * fdx_op->right_len) * num_of_vars;

      mympi->pair_siz_rbuff_y1[ipair][istage] = (ni * nk * fdy_op->left_len ) * num_of_vars;
      mympi->pair_siz_rbuff_y2[ipair][istage] = (ni * nk * fdy_op->right_len) * num_of_vars;

      size_t siz_s =  mympi->pair_siz_sbuff_x1[ipair][istage]
                    + mympi->pair_siz_sbuff_x2[ipair][istage]
                    + mympi->pair_siz_sbuff_y1[ipair][istage]
                    + mympi->pair_siz_sbuff_y2[ipair][istage];
      size_t siz_r =  mympi->pair_siz_rbuff_x1[ipair][istage]
                    + mympi->pair_siz_rbuff_x2[ipair][istage]
                    + mympi->pair_siz_rbuff_y1[ipair][istage]
                    + mympi->pair_siz_rbuff_y2[ipair][istage];

      if (siz_s > mympi->siz_sbuff) mympi->siz_sbuff = siz_s;
      if (siz_r > mympi->siz_rbuff) mympi->siz_rbuff = siz_r;
    }
  }
  // alloc in gpu
  mympi->sbuff = (float *) cuda_malloc(mympi->siz_sbuff * sizeof(MPI_FLOAT));
  mympi->rbuff = (float *) cuda_malloc(mympi->siz_rbuff * sizeof(MPI_FLOAT));
  // set up pers communication
  for (int ipair = 0; ipair < fd->num_of_pairs; ipair++)
  {
    for (int istage = 0; istage < fd->num_rk_stages; istage++)
    {
      size_t siz_s_x1 = mympi->pair_siz_sbuff_x1[ipair][istage];
      size_t siz_s_x2 = mympi->pair_siz_sbuff_x2[ipair][istage];
      size_t siz_s_y1 = mympi->pair_siz_sbuff_y1[ipair][istage];
      size_t siz_s_y2 = mympi->pair_siz_sbuff_y2[ipair][istage];

      float *sbuff_x1 = mympi->sbuff;
      float *sbuff_x2 = sbuff_x1 + siz_s_x1;
      float *sbuff_y1 = sbuff_x2 + siz_s_x2;
      float *sbuff_y2 = sbuff_y1 + siz_s_y1;

      // npair: xx, nstage: x, 
      int tag_pair_stage = ipair * 1000 + istage * 100;
      int tag[4] = { tag_pair_stage+11, tag_pair_stage+12, tag_pair_stage+21, tag_pair_stage+22 };

      // send
      MPI_Send_init(sbuff_x1, siz_s_x1, MPI_FLOAT, mympi->neighid[0], tag[0], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][0]));
      MPI_Send_init(sbuff_x2, siz_s_x2, MPI_FLOAT, mympi->neighid[1], tag[1], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][1]));
      MPI_Send_init(sbuff_y1, siz_s_y1, MPI_FLOAT, mympi->neighid[2], tag[2], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][2]));
      MPI_Send_init(sbuff_y2, siz_s_y2, MPI_FLOAT, mympi->neighid[3], tag[3], mympi->topocomm, &(mympi->pair_s_reqs[ipair][istage][3]));

      // recv
      size_t siz_r_x1 = mympi->pair_siz_rbuff_x1[ipair][istage];
      size_t siz_r_x2 = mympi->pair_siz_rbuff_x2[ipair][istage];
      size_t siz_r_y1 = mympi->pair_siz_rbuff_y1[ipair][istage];
      size_t siz_r_y2 = mympi->pair_siz_rbuff_y2[ipair][istage];

      float *rbuff_x1 = mympi->rbuff;
      float *rbuff_x2 = rbuff_x1 + siz_r_x1;
      float *rbuff_y1 = rbuff_x2 + siz_r_x2;
      float *rbuff_y2 = rbuff_y1 + siz_r_y1;

      // recv
      MPI_Recv_init(rbuff_x1, siz_r_x1, MPI_FLOAT, mympi->neighid[0], tag[1], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][0]));
      MPI_Recv_init(rbuff_x2, siz_r_x2, MPI_FLOAT, mympi->neighid[1], tag[0], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][1]));
      MPI_Recv_init(rbuff_y1, siz_r_y1, MPI_FLOAT, mympi->neighid[2], tag[3], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][2]));
      MPI_Recv_init(rbuff_y2, siz_r_y2, MPI_FLOAT, mympi->neighid[3], tag[2], mympi->topocomm, &(mympi->pair_r_reqs[ipair][istage][3]));
    }
  }

  return;
}

int 
blk_macdrp_pack_mesg_gpu(float * w_cur,
                         fd_t *fd,
                         gdinfo_t *gdinfo, 
                         mympi_t *mympi, 
                         int ipair_mpi,
                         int istage_mpi,
                         int myid)
{
  //
  //nx1_g is fdx_op->right_len;
  //nx2_g is fdx_op->left_len;
  //ny1_g is fdy_op->right_len;
  //ny2_g is fdy_op->left_len;
  //
  int ni1 = gdinfo->ni1;
  int ni2 = gdinfo->ni2;
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  size_t siz_iy   = gdinfo->siz_iy;
  size_t siz_iz  = gdinfo->siz_iz;
  size_t siz_icmp = gdinfo->siz_icmp;
  int ni = ni2-ni1+1;
  int nj = nj2-nj1+1;
  int nk = nk2-nk1+1;

  // ghost point
  int nx1_g = fd->pair_fdx_op[ipair_mpi][istage_mpi][fd->num_of_fdx_op-1].right_len;
  int nx2_g = fd->pair_fdx_op[ipair_mpi][istage_mpi][fd->num_of_fdx_op-1].left_len;
  int ny1_g = fd->pair_fdy_op[ipair_mpi][istage_mpi][fd->num_of_fdy_op-1].right_len;
  int ny2_g = fd->pair_fdy_op[ipair_mpi][istage_mpi][fd->num_of_fdy_op-1].left_len;
  size_t siz_sbuff_x1 = mympi->pair_siz_sbuff_x1[ipair_mpi][istage_mpi];
  size_t siz_sbuff_x2 = mympi->pair_siz_sbuff_x2[ipair_mpi][istage_mpi];
  size_t siz_sbuff_y1 = mympi->pair_siz_sbuff_y1[ipair_mpi][istage_mpi];
  
  float *sbuff_x1 = mympi->sbuff;
  float *sbuff_x2 = sbuff_x1 + siz_sbuff_x1;
  float *sbuff_y1 = sbuff_x2 + siz_sbuff_x2;
  float *sbuff_y2 = sbuff_y1 + siz_sbuff_y1;

  {
    dim3 block(nx1_g,8,8);
    dim3 grid;
    grid.x = (nx1_g + block.x -1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_x1<<<grid, block >>>(
           w_cur, sbuff_x1, siz_iy, siz_iz, siz_icmp,
           ni1, nj1, nk1, nx1_g, nj, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(nx2_g,8,8);
    dim3 grid;
    grid.x = (nx2_g + block.x -1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_x2<<<grid, block >>>(
           w_cur, sbuff_x2, siz_iy, siz_iz, siz_icmp,
           ni2, nj1, nk1, nx2_g, nj, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny1_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny1_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_y1<<<grid, block >>>(
           w_cur, sbuff_y1, siz_iy, siz_iz, siz_icmp,
           ni1, nj1, nk1, ni, ny1_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny2_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny2_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_pack_mesg_y2<<<grid, block >>>(
           w_cur, sbuff_y2, siz_iy, siz_iz, siz_icmp,
           ni1, nj2, nk1, ni, ny2_g, nk);
    CUDACHECK(cudaDeviceSynchronize());
  }

  return 0;
}

__global__ void
blk_macdrp_pack_mesg_x1(
           float *w_cur, float *sbuff_x1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj1, int nk1, int nx1_g, int nj, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<nx1_g && iy<nj && iz<nk)
  {
    iptr     = (iz+nk1) * siz_iz + (iy+nj1) * siz_iy + (ix+ni1);
    iptr_b   = iz*nj*nx1_g + iy*nx1_g + ix;
    sbuff_x1[iptr_b + 0*nx1_g*nj*nk] = w_cur[iptr + 0*siz_icmp];
    sbuff_x1[iptr_b + 1*nx1_g*nj*nk] = w_cur[iptr + 1*siz_icmp];
    sbuff_x1[iptr_b + 2*nx1_g*nj*nk] = w_cur[iptr + 2*siz_icmp];
    sbuff_x1[iptr_b + 3*nx1_g*nj*nk] = w_cur[iptr + 3*siz_icmp];
    sbuff_x1[iptr_b + 4*nx1_g*nj*nk] = w_cur[iptr + 4*siz_icmp];
    sbuff_x1[iptr_b + 5*nx1_g*nj*nk] = w_cur[iptr + 5*siz_icmp];
    sbuff_x1[iptr_b + 6*nx1_g*nj*nk] = w_cur[iptr + 6*siz_icmp];
    sbuff_x1[iptr_b + 7*nx1_g*nj*nk] = w_cur[iptr + 7*siz_icmp];
    sbuff_x1[iptr_b + 8*nx1_g*nj*nk] = w_cur[iptr + 8*siz_icmp];
  }
  return;
}

__global__ void
blk_macdrp_pack_mesg_x2(
           float *w_cur, float *sbuff_x2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni2, int nj1, int nk1, int nx2_g, int nj, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<nx2_g && iy<nj && iz<nk)
  {
    iptr     = (iz+nk1) * siz_iz + (iy+nj1) * siz_iy + (ix+ni2-nx2_g+1);
    iptr_b   = iz*nj*nx2_g + iy*nx2_g + ix;
    sbuff_x2[iptr_b + 0*nx2_g*nj*nk] = w_cur[iptr + 0*siz_icmp];
    sbuff_x2[iptr_b + 1*nx2_g*nj*nk] = w_cur[iptr + 1*siz_icmp];
    sbuff_x2[iptr_b + 2*nx2_g*nj*nk] = w_cur[iptr + 2*siz_icmp];
    sbuff_x2[iptr_b + 3*nx2_g*nj*nk] = w_cur[iptr + 3*siz_icmp];
    sbuff_x2[iptr_b + 4*nx2_g*nj*nk] = w_cur[iptr + 4*siz_icmp];
    sbuff_x2[iptr_b + 5*nx2_g*nj*nk] = w_cur[iptr + 5*siz_icmp];
    sbuff_x2[iptr_b + 6*nx2_g*nj*nk] = w_cur[iptr + 6*siz_icmp];
    sbuff_x2[iptr_b + 7*nx2_g*nj*nk] = w_cur[iptr + 7*siz_icmp];
    sbuff_x2[iptr_b + 8*nx2_g*nj*nk] = w_cur[iptr + 8*siz_icmp];
  }
  return;
}

__global__ void
blk_macdrp_pack_mesg_y1(
           float *w_cur, float *sbuff_y1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj1, int nk1, int ni, int ny1_g, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<ny1_g && iz<nk)
  {
    iptr     = (iz+nk1) * siz_iz + (iy+nj1) * siz_iy + (ix+ni1);
    iptr_b   = iz*ni*ny1_g + iy*ni + ix;
    sbuff_y1[iptr_b + 0*ny1_g*ni*nk] = w_cur[iptr + 0*siz_icmp];
    sbuff_y1[iptr_b + 1*ny1_g*ni*nk] = w_cur[iptr + 1*siz_icmp];
    sbuff_y1[iptr_b + 2*ny1_g*ni*nk] = w_cur[iptr + 2*siz_icmp];
    sbuff_y1[iptr_b + 3*ny1_g*ni*nk] = w_cur[iptr + 3*siz_icmp];
    sbuff_y1[iptr_b + 4*ny1_g*ni*nk] = w_cur[iptr + 4*siz_icmp];
    sbuff_y1[iptr_b + 5*ny1_g*ni*nk] = w_cur[iptr + 5*siz_icmp];
    sbuff_y1[iptr_b + 6*ny1_g*ni*nk] = w_cur[iptr + 6*siz_icmp];
    sbuff_y1[iptr_b + 7*ny1_g*ni*nk] = w_cur[iptr + 7*siz_icmp];
    sbuff_y1[iptr_b + 8*ny1_g*ni*nk] = w_cur[iptr + 8*siz_icmp];
  }

  return;
}

__global__ void
blk_macdrp_pack_mesg_y2(
           float *w_cur, float *sbuff_y2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj2, int nk1, int ni, int ny2_g, int nk)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if(ix<ni && iy<ny2_g && iz<nk)
  {
    iptr     = (iz+nk1) * siz_iz + (iy+nj2-ny2_g+1) * siz_iy + (ix+ni1);
    iptr_b   = iz*ni*ny2_g + iy*ni + ix;
    sbuff_y2[iptr_b + 0*ny2_g*ni*nk] = w_cur[iptr + 0*siz_icmp];
    sbuff_y2[iptr_b + 1*ny2_g*ni*nk] = w_cur[iptr + 1*siz_icmp];
    sbuff_y2[iptr_b + 2*ny2_g*ni*nk] = w_cur[iptr + 2*siz_icmp];
    sbuff_y2[iptr_b + 3*ny2_g*ni*nk] = w_cur[iptr + 3*siz_icmp];
    sbuff_y2[iptr_b + 4*ny2_g*ni*nk] = w_cur[iptr + 4*siz_icmp];
    sbuff_y2[iptr_b + 5*ny2_g*ni*nk] = w_cur[iptr + 5*siz_icmp];
    sbuff_y2[iptr_b + 6*ny2_g*ni*nk] = w_cur[iptr + 6*siz_icmp];
    sbuff_y2[iptr_b + 7*ny2_g*ni*nk] = w_cur[iptr + 7*siz_icmp];
    sbuff_y2[iptr_b + 8*ny2_g*ni*nk] = w_cur[iptr + 8*siz_icmp];
  }
  return;
}

int 
blk_macdrp_unpack_mesg_gpu(float *w_cur, 
                           fd_t *fd,
                           gdinfo_t *gdinfo,
                           mympi_t *mympi, 
                           int ipair_mpi,
                           int istage_mpi,
                           int *neighid)
{
  //
  //nx1_g is fdx_op->right_len;
  //nx2_g is fdx_op->left_len;
  //ny1_g is fdy_op->right_len;
  //ny2_g is fdy_op->left_len;
  //
  int ni1 = gdinfo->ni1;
  int ni2 = gdinfo->ni2;
  int nj1 = gdinfo->nj1;
  int nj2 = gdinfo->nj2;
  int nk1 = gdinfo->nk1;
  int nk2 = gdinfo->nk2;
  size_t siz_iy   = gdinfo->siz_iy;
  size_t siz_iz  = gdinfo->siz_iz;
  size_t siz_icmp = gdinfo->siz_icmp;

  int ni = ni2-ni1+1;
  int nj = nj2-nj1+1;
  int nk = nk2-nk1+1;
  
  // ghost point
  int nx1_g = fd->pair_fdx_op[ipair_mpi][istage_mpi][fd->num_of_fdx_op-1].right_len;
  int nx2_g = fd->pair_fdx_op[ipair_mpi][istage_mpi][fd->num_of_fdx_op-1].left_len;
  int ny1_g = fd->pair_fdy_op[ipair_mpi][istage_mpi][fd->num_of_fdy_op-1].right_len;
  int ny2_g = fd->pair_fdy_op[ipair_mpi][istage_mpi][fd->num_of_fdy_op-1].left_len;
  size_t siz_rbuff_x1 = mympi->pair_siz_rbuff_x1[ipair_mpi][istage_mpi];
  size_t siz_rbuff_x2 = mympi->pair_siz_rbuff_x2[ipair_mpi][istage_mpi];
  size_t siz_rbuff_y1 = mympi->pair_siz_rbuff_y1[ipair_mpi][istage_mpi];
  float *rbuff_x1 = mympi->rbuff;
  float *rbuff_x2 = rbuff_x1 + siz_rbuff_x1;
  float *rbuff_y1 = rbuff_x2 + siz_rbuff_x2;
  float *rbuff_y2 = rbuff_y1 + siz_rbuff_y1;
  {
    dim3 block(nx2_g,8,8);
    dim3 grid;
    grid.x = (nx2_g + block.x -1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_x1<<< grid, block >>>(
           w_cur, rbuff_x1, siz_iy, siz_iz, siz_icmp,
           ni1, nj1, nk1, nx2_g, nj, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(nx1_g,8,8);
    dim3 grid;
    grid.x = (nx1_g + block.x -1) / block.x;
    grid.y = (nj + block.y - 1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_x2<<< grid, block >>>(
           w_cur, rbuff_x2, siz_iy, siz_iz, siz_icmp,
           ni2, nj1, nk1, nx1_g, nj, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny2_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny2_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_y1<<< grid, block >>>(
           w_cur, rbuff_y1, siz_iy, siz_iz, siz_icmp,
           ni1, nj1, nk1, ni, ny2_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  {
    dim3 block(8,ny1_g,8);
    dim3 grid;
    grid.x = (ni + block.x - 1) / block.x;
    grid.y = (ny1_g + block.y -1) / block.y;
    grid.z = (nk + block.z - 1) / block.z;
    blk_macdrp_unpack_mesg_y2<<< grid, block >>>(
           w_cur, rbuff_y2, siz_iy, siz_iz, siz_icmp,
           ni1, nj2, nk1, ni, ny1_g, nk, neighid);
    CUDACHECK(cudaDeviceSynchronize());
  }
  return 0;
}

//from x2
__global__ void
blk_macdrp_unpack_mesg_x1(
           float *w_cur, float *rbuff_x1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj1, int nk1, int nx2_g, int nj, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[0] != MPI_PROC_NULL) {
    if(ix<nx2_g && iy<nj && iz<nk){
      iptr   = (iz+nk1) * siz_iz + (iy+nj1) * siz_iy + (ix+ni1-nx2_g);
      iptr_b = iz*nj*nx2_g + iy*nx2_g + ix;
      w_cur[iptr + 0*siz_icmp] = rbuff_x1[iptr_b+ 0*nx2_g*nj*nk];
      w_cur[iptr + 1*siz_icmp] = rbuff_x1[iptr_b+ 1*nx2_g*nj*nk];
      w_cur[iptr + 2*siz_icmp] = rbuff_x1[iptr_b+ 2*nx2_g*nj*nk];
      w_cur[iptr + 3*siz_icmp] = rbuff_x1[iptr_b+ 3*nx2_g*nj*nk];
      w_cur[iptr + 4*siz_icmp] = rbuff_x1[iptr_b+ 4*nx2_g*nj*nk];
      w_cur[iptr + 5*siz_icmp] = rbuff_x1[iptr_b+ 5*nx2_g*nj*nk];
      w_cur[iptr + 6*siz_icmp] = rbuff_x1[iptr_b+ 6*nx2_g*nj*nk];
      w_cur[iptr + 7*siz_icmp] = rbuff_x1[iptr_b+ 7*nx2_g*nj*nk];
      w_cur[iptr + 8*siz_icmp] = rbuff_x1[iptr_b+ 8*nx2_g*nj*nk];
    }
  }
  return;
}

//from x1
__global__ void
blk_macdrp_unpack_mesg_x2(
           float *w_cur, float *rbuff_x2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni2, int nj1, int nk1, int nx1_g, int nj, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[1] != MPI_PROC_NULL) {
    if(ix<nx1_g && iy<nj && iz<nk){
      iptr   = (iz+nk1) * siz_iz + (iy+nj1) * siz_iy + (ix+ni2+1);
      iptr_b = iz*nj*nx1_g + iy*nx1_g + ix;
      w_cur[iptr + 0*siz_icmp] = rbuff_x2[iptr_b+ 0*nx1_g*nj*nk];
      w_cur[iptr + 1*siz_icmp] = rbuff_x2[iptr_b+ 1*nx1_g*nj*nk];
      w_cur[iptr + 2*siz_icmp] = rbuff_x2[iptr_b+ 2*nx1_g*nj*nk];
      w_cur[iptr + 3*siz_icmp] = rbuff_x2[iptr_b+ 3*nx1_g*nj*nk];
      w_cur[iptr + 4*siz_icmp] = rbuff_x2[iptr_b+ 4*nx1_g*nj*nk];
      w_cur[iptr + 5*siz_icmp] = rbuff_x2[iptr_b+ 5*nx1_g*nj*nk];
      w_cur[iptr + 6*siz_icmp] = rbuff_x2[iptr_b+ 6*nx1_g*nj*nk];
      w_cur[iptr + 7*siz_icmp] = rbuff_x2[iptr_b+ 7*nx1_g*nj*nk];
      w_cur[iptr + 8*siz_icmp] = rbuff_x2[iptr_b+ 8*nx1_g*nj*nk];
    }
  }
  return;
}

//from y2
__global__ void
blk_macdrp_unpack_mesg_y1(
           float *w_cur, float *rbuff_y1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj1, int nk1, int ni, int ny2_g, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[2] != MPI_PROC_NULL) {
    if(ix<ni && iy<ny2_g && iz<nk){
      iptr   = (iz+nk1) * siz_iz + (iy+nj1-ny2_g) * siz_iy + (ix+ni1);
      iptr_b = iz*ni*ny2_g + iy*ni + ix;
      w_cur[iptr + 0*siz_icmp] = rbuff_y1[iptr_b+ 0*ny2_g*ni*nk];
      w_cur[iptr + 1*siz_icmp] = rbuff_y1[iptr_b+ 1*ny2_g*ni*nk];
      w_cur[iptr + 2*siz_icmp] = rbuff_y1[iptr_b+ 2*ny2_g*ni*nk];
      w_cur[iptr + 3*siz_icmp] = rbuff_y1[iptr_b+ 3*ny2_g*ni*nk];
      w_cur[iptr + 4*siz_icmp] = rbuff_y1[iptr_b+ 4*ny2_g*ni*nk];
      w_cur[iptr + 5*siz_icmp] = rbuff_y1[iptr_b+ 5*ny2_g*ni*nk];
      w_cur[iptr + 6*siz_icmp] = rbuff_y1[iptr_b+ 6*ny2_g*ni*nk];
      w_cur[iptr + 7*siz_icmp] = rbuff_y1[iptr_b+ 7*ny2_g*ni*nk];
      w_cur[iptr + 8*siz_icmp] = rbuff_y1[iptr_b+ 8*ny2_g*ni*nk];
    }
  }
  return;
}

//from y1
__global__ void
blk_macdrp_unpack_mesg_y2(
           float *w_cur, float *rbuff_y2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int ni1, int nj2, int nk1, int ni, int ny1_g, int nk, int *neighid)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  size_t iptr_b;
  size_t iptr;
  if (neighid[3] != MPI_PROC_NULL) {
    if(ix<ni && iy<ny1_g && iz<nk){
      iptr   = (iz+nk1) * siz_iz + (iy+nj2+1) * siz_iy + (ix+ni1);
      iptr_b = iz*ni*ny1_g + iy*ni + ix;
      w_cur[iptr + 0*siz_icmp] = rbuff_y2[iptr_b+ 0*ny1_g*ni*nk];
      w_cur[iptr + 1*siz_icmp] = rbuff_y2[iptr_b+ 1*ny1_g*ni*nk];
      w_cur[iptr + 2*siz_icmp] = rbuff_y2[iptr_b+ 2*ny1_g*ni*nk];
      w_cur[iptr + 3*siz_icmp] = rbuff_y2[iptr_b+ 3*ny1_g*ni*nk];
      w_cur[iptr + 4*siz_icmp] = rbuff_y2[iptr_b+ 4*ny1_g*ni*nk];
      w_cur[iptr + 5*siz_icmp] = rbuff_y2[iptr_b+ 5*ny1_g*ni*nk];
      w_cur[iptr + 6*siz_icmp] = rbuff_y2[iptr_b+ 6*ny1_g*ni*nk];
      w_cur[iptr + 7*siz_icmp] = rbuff_y2[iptr_b+ 7*ny1_g*ni*nk];
      w_cur[iptr + 8*siz_icmp] = rbuff_y2[iptr_b+ 8*ny1_g*ni*nk];
    }
  }
  return;
}

/*********************************************************************
 * estimate dt
 *********************************************************************/

int
blk_dt_esti_curv(gdinfo_t *gdinfo, gd_t *gdcurv, md_t *md,
    float CFL, float *dtmax, float *dtmaxVp, float *dtmaxL,
    int *dtmaxi, int *dtmaxj, int *dtmaxk)
{
  int ierr = 0;

  float dtmax_local = 1.0e10;
  float Vp;

  float *x3d = gdcurv->x3d;
  float *y3d = gdcurv->y3d;
  float *z3d = gdcurv->z3d;

  for (int k = gdinfo->nk1; k < gdinfo->nk2; k++)
  {
    for (int j = gdinfo->nj1; j < gdinfo->nj2; j++)
    {
      for (int i = gdinfo->ni1; i < gdinfo->ni2; i++)
      {
        size_t iptr = i + j * gdinfo->siz_iy + k * gdinfo->siz_iz;

        if (md->medium_type == CONST_MEDIUM_ELASTIC_ISO) {
          Vp = sqrt( (md->lambda[iptr] + 2.0 * md->mu[iptr]) / md->rho[iptr] );
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_VTI) {
          float Vpv = sqrt( md->c33[iptr] / md->rho[iptr] );
          float Vph = sqrt( md->c11[iptr] / md->rho[iptr] );
          Vp = Vph > Vpv ? Vph : Vpv;
        } else if (md->medium_type == CONST_MEDIUM_ELASTIC_ANISO) {
          // need to implement accurate solution
          Vp = sqrt( md->c11[iptr] / md->rho[iptr] );
        } else if (md->medium_type == CONST_MEDIUM_ACOUSTIC_ISO) {
          Vp = sqrt( md->kappa[iptr] / md->rho[iptr] );
        }

        float dtLe = 1.0e20;
        float p0[] = { x3d[iptr], y3d[iptr], z3d[iptr] };

        // min L to 8 adjacent planes
        for (int kk = -1; kk <=1; kk++) {
          for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
              if (ii != 0 && jj !=0 && kk != 0)
              {
                float p1[] = { x3d[iptr-ii], y3d[iptr-ii], z3d[iptr-ii] };
                float p2[] = { x3d[iptr-jj*gdinfo->siz_iy],
                               y3d[iptr-jj*gdinfo->siz_iy],
                               z3d[iptr-jj*gdinfo->siz_iy] };
                float p3[] = { x3d[iptr-kk*gdinfo->siz_iz],
                               y3d[iptr-kk*gdinfo->siz_iz],
                               z3d[iptr-kk*gdinfo->siz_iz] };

                float L = fdlib_math_dist_point2plane(p0, p1, p2, p3);

                if (dtLe > L) dtLe = L;
              }
            }
          }
        }

        // convert to dt
        float dt_point = CFL / Vp * dtLe;

        // if smaller
        if (dt_point < dtmax_local) {
          dtmax_local = dt_point;
          *dtmaxi = i;
          *dtmaxj = j;
          *dtmaxk = k;
          *dtmaxVp = Vp;
          *dtmaxL  = dtLe;
        }

      } // i
    } // i
  } //k

  *dtmax = dtmax_local;

  return ierr;
}

float
blk_keep_two_digi(float dt)
{
  char str[40];
  float dt_2;

  sprintf(str, "%6.4e", dt);

  str[3] = '0';
  str[4] = '0';
  str[5] = '0';

  sscanf(str, "%f", &dt_2);
  
  return dt_2;
}
