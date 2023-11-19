function [V, X, Y, Z, t] = gather_slice_y(output_dir,nlayer,varnm,sliceid,nproi,nprok)



  for kp=0:nprok-1
    for ip=0:nproi-1
      % snapshot data
      slicestruct=dir([output_dir,'/','slicey_j',num2str(sliceid),'_px',num2str(ip),'_py*','_pz',num2str(kp),'.nc']);
      slicenm=slicestruct.name;
      slicenm_dir=[output_dir,'/',slicenm];
      pnistruct=nc_getdiminfo(slicenm_dir,'i');
      pni=pnistruct.Length;
      pnkstruct=nc_getdiminfo(slicenm_dir,'k');
      pnk=pnkstruct.Length;
      if ip==0
          VV=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnk,pni],[1,1,1]));
      else
          VV0=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnk,pni],[1,1,1]));
          VV=vertcat(VV,VV0);
      end
      t=nc_varget(slicenm_dir,'time',[nlayer-1],[1]);
      
      % coordinate data
      jp=str2num(slicenm( strfind(slicenm,'py')+2 : strfind(slicenm,'_pz')-1 ));
      coordnm=['coord','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      coordnm_dir=[output_dir,'/',coordnm];
      coorddimstruct=nc_getdiminfo(coordnm_dir,'i');
      slicedimstruct=nc_getdiminfo(slicenm_dir,'i');
      ghostp=(coorddimstruct.Length-slicedimstruct.Length)/2;
      % delete 3 ghost points
      j_index=double(nc_attget(slicenm_dir,nc_global,'j_index_with_ghosts_in_this_thread')) - 3;
      if ip==0
        XX=squeeze(nc_varget(coordnm_dir,'x',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
        YY=squeeze(nc_varget(coordnm_dir,'y',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
        ZZ=squeeze(nc_varget(coordnm_dir,'z',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
      else
        XX0=squeeze(nc_varget(coordnm_dir,'x',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
        YY0=squeeze(nc_varget(coordnm_dir,'y',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
        ZZ0=squeeze(nc_varget(coordnm_dir,'z',[ghostp,j_index,ghostp],[pnk,1,pni],[1,1,1]));
        XX=vertcat(XX,XX0);
        YY=vertcat(YY,YY0);
        ZZ=vertcat(ZZ,ZZ0);
      end  
    end % end ip
    if kp==0
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
  end % end kp

end  % end function
