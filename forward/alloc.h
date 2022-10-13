#ifndef ALLOC_H
#define ALLOC_H

#include "constants.h"
#include "gd_t.h"
#include "fd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "src_t.h"
#include "bdry_t.h"

int
init_gdinfo_device(gdinfo_t *gdinfo, gdinfo_t *gdinfo_d);

int init_gdcart_device(gd_t *gdcart, gd_t *gdcart_d);

int init_gdcurv_device(gd_t *gdcurv, gd_t *gdcurv_d);

int
init_md_device(md_t *md, md_t *md_d);

int
init_fd_device(fd_t *fd, fd_wav_t *fd_wav_d);

int
init_metric_device(gdcurv_metric_t *metric, gdcurv_metric_t *metric_d);

int
init_src_device(src_t *src, src_t *src_d);

int 
init_bdryfree_device(gdinfo_t *gdinfo, bdryfree_t *bdryfree, bdryfree_t *bdryfree_d);

int 
init_bdrypml_device(gdinfo_t *gdinfo, bdrypml_t *bdrypml, bdrypml_t *bdrypml_d);

int
init_bdryexp_device(gdinfo_t *gdinfo, bdryexp_t *bdryexp, bdryexp_t *bdryexp_d);

int 
init_wave_device(wav_t *wav, wav_t *wav_d);

float *
init_PGVAD_device(gdinfo_t *gdinfo);

float *
init_Dis_accu_device(gdinfo_t *gdinfo);

int *
init_neighid_device(int *neighid);

int 
dealloc_gdcurv_device(gd_t gdcurv_d);

int 
dealloc_md_device(md_t md_d);

int 
dealloc_fd_device(fd_wav_t fd_wav_d);

int
dealloc_metric_device(gdcurv_metric_t metric_d);

int 
dealloc_src_device(src_t src_d);

int
dealloc_bdryfree_device(bdryfree_t bdryfree_d);

int 
dealloc_bdrypml_device(bdrypml_t bdrypml_d);

int
dealloc_bdryexp_device(bdryexp_t bdryexp_d);

int 
dealloc_wave_device(wav_t wav_d);

#endif
