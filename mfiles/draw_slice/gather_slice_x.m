function [V, X, Y, Z, t] = gather_slice_x(output_dir,nlayer,varnm,sliceid,nproj,nprok)

  for kp=0:nprok-1      
    for jp=0:nproj-1
      % snapshot data
      slicestruct=dir([output_dir,'/','slicex_i',num2str(sliceid),'_px*_py',num2str(jp),'_pz',num2str(kp),'.nc']);
      slicenm=slicestruct.name;
      slicenm_dir=[output_dir,'/',slicenm];
      pnjstruct=nc_getdiminfo(slicenm_dir,'j');
      pnj=pnjstruct.Length;
      pnkstruct=nc_getdiminfo(slicenm_dir,'k');
      pnk=pnkstruct.Length;
      if jp==0
          VV=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnk,pnj],[1,1,1]));
      else
          VV0=squeeze(nc_varget(slicenm_dir,varnm,[nlayer-1,0,0],[1,pnk,pnj],[1,1,1]));
          VV=horzcat(VV,VV0);
      end
      t=nc_varget(slicenm_dir,'time',[nlayer-1],[1]);
      
      % coordinate data
      ip=str2num(slicenm( strfind(slicenm,'px')+2 : strfind(slicenm,'_py')-1 ));
      coordnm=['coord','_px',num2str(ip),'_py',num2str(jp),'_pz',num2str(kp),'.nc'];
      coordnm_dir=[output_dir,'/',coordnm];
      coorddimstruct=nc_getdiminfo(coordnm_dir,'k');
      slicedimstruct=nc_getdiminfo(slicenm_dir,'k');
      ghostp=(coorddimstruct.Length-slicedimstruct.Length)/2;
      % delete 3 ghost points
      i_index=double(nc_attget(slicenm_dir,nc_global,'i_index_with_ghosts_in_this_thread')) - 3;
      if jp==0
          XX=squeeze(nc_varget(coordnm_dir,'x',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
          YY=squeeze(nc_varget(coordnm_dir,'y',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
          ZZ=squeeze(nc_varget(coordnm_dir,'z',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
      else
          XX0=squeeze(nc_varget(coordnm_dir,'x',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
          YY0=squeeze(nc_varget(coordnm_dir,'y',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
          ZZ0=squeeze(nc_varget(coordnm_dir,'z',[ghostp,ghostp,i_index],[pnk,pnj,1],[1,1,1]));
          XX=horzcat(XX,XX0);
          YY=horzcat(YY,YY0);
          ZZ=horzcat(ZZ,ZZ0);
      end             
    end % end jp
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
  end  % end kp

end % end function
