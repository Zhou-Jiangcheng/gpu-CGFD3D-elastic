function [v,t] =gather_snap(snapinfo,nlayer,varnm,snap_dir)

% check dir exists
if ~ exist(snap_dir,'dir')
    error([mfilename ': directory ' snap_dir ' does not exist']);
end

% load
nthd=length(snapinfo);
for n=1:nthd
    
    n_i=snapinfo(n).thisid(1); n_j=snapinfo(n).thisid(2);n_k=snapinfo(n).thisid(3);
    i1=snapinfo(n).indxs(1); j1=snapinfo(n).indxs(2); k1=snapinfo(n).indxs(3);
    i2=snapinfo(n).indxe(1); j2=snapinfo(n).indxe(2); k2=snapinfo(n).indxe(3);
    subs=snapinfo(n).subs;
    subc=snapinfo(n).subc;
    subt=snapinfo(n).subt;
    fnm_snap=[snap_dir,'/',snapinfo(n).fnmprefix,'_px',num2str(n_i),...
              '_py',num2str(n_j),'_pz',num2str(n_k),'.nc'];
    if ~ exist(fnm_snap)
       error([mfilename ': file ',fnm_snap, ' does not exist']);
    end
    tdim=nc_getdiminfo(fnm_snap,'time');
    if tdim.Length==0 | (nlayer-1)-1>=tdim.Length
       error([num2str(nlayer) 'th layer is beyond current time dim (' ...
            num2str(tdim.Length) ') in ' fnm_snap]);
    end
    subs = fliplr(subs); subc = fliplr(subc); subt = fliplr(subt);

    % get data
    v(k1:k2,j1:j2,i1:i2)=nc_varget(fnm_snap,varnm, ...
          [nlayer-1,subs-1],[1,subc],[1,subt]);
    t=nc_varget(fnm_snap,'time',[nlayer-1],[1]);

end
v=squeeze(v);

end
