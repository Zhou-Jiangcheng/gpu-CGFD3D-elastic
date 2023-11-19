function [V, X, Y, Z, t] = gather_slice_z(output_dir,nlayer,varnm,sliceid,nproi,nproj)



  for jp=0:nproj-1
    for ip=0:nproi-1
      % snapshot data
      slicestruct=dir([output_dir,'/','slicez_k',num2str(sliceid),'_px',num2str(ip),'_py',num2str(jp),'_pz*.nc']);
      slicenm=slicestruct.name;
      slicenm_dir=[output_dir,'/',slicenm];
      pnistruct=nc_getdiminfo(slicenm_dir,'i');
      pni=pnistruct.Length;
      pnjstruct=nc_getdiminfo(slicenm_dir,'j');
      pnj=pnjstruct.Length;
      if ip==0
          VV=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnj,pni],[1,1,1]));
      else
          VV0=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnj,pni],[1,1,1]));
          VV=horzcat(VV,VV0);
      end
      t=nc_varget(slicenm_dir,'time',[nlayer-1],[1]);
      
      % coordinate data
      kp=str2num(slicenm( strfind(slicenm,'pz')+2 : strfind(slicenm,'.nc')-1 ));
      coordnm=['coord','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      coordnm_dir=[output_dir,'/',coordnm];
      coorddimstruct=nc_getdiminfo(coordnm_dir,'j');
      slicedimstruct=nc_getdiminfo(slicenm_dir,'j');
      ghostp=(coorddimstruct.Length-slicedimstruct.Length)/2;
      % delete 3 ghost points
      k_index=double(nc_attget(slicenm_dir,nc_global,'k_index_with_ghosts_in_this_thread')) - 3;
      if ip==0
        XX=squeeze(nc_varget(coordnm_dir,'x',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
        YY=squeeze(nc_varget(coordnm_dir,'y',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
        ZZ=squeeze(nc_varget(coordnm_dir,'z',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
      else
        XX0=squeeze(nc_varget(coordnm_dir,'x',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
        YY0=squeeze(nc_varget(coordnm_dir,'y',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
        ZZ0=squeeze(nc_varget(coordnm_dir,'z',[k_index,ghostp,ghostp],[1,pnj,pni],[1,1,1]));
        XX=horzcat(XX,XX0);
        YY=horzcat(YY,YY0);
        ZZ=horzcat(ZZ,ZZ0);
      end  
    end % end ip
    if jp==0
      V=VV;
      X=XX;
      Y=YY;
      Z=ZZ;
    else
      V=vertcat(V,VV);
      X=vertcat(X,XX);
      Y=vertcat(Y,YY);
      Z=vertcat(Z,ZZ);
    end
  end % end jp 

end % end function
