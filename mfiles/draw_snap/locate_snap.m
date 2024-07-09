function snapinfo = locate_snap(parfnm,snap_dir,id,subs,subc,subt)

% check parameter file exist
if ~ exist(parfnm,'file')
    error([mfilename ': file ' parfnm ' does not exist']);
end

% read parameters file
par=loadjson(parfnm);
snap_subs=par.snapshot{id}.grid_index_start;
snap_subc=par.snapshot{id}.grid_index_count;
snap_subt=par.snapshot{id}.grid_index_incre;
snap_tinv=par.snapshot{id}.time_index_incre;

gsubs = subs;
gsubt = subt;
gsubc = subc;

% reset count=-1 to total number
indx=find(subc==-1);
gsubc(indx)=ceil((snap_subc(indx)-gsubs(indx)+1)./gsubt(indx));
gsube=gsubs+(gsubc-1).*gsubt;

% search the nc file headers to locate the threads/processors
snapprefix=par.snapshot{id}.name;
snaplist=dir([snap_dir,'/',snapprefix,'*.nc']);
n=0;
for i=1:length(snaplist)
    
    snapnm=[snap_dir,'/',snaplist(i).name];
    xyzs=double(nc_attget(snapnm,nc_global,'first_index_to_snapshot_output'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xc=nc_getdiminfo(snapnm,'i');
    xc=xc.Length;
    yc=nc_getdiminfo(snapnm,'j');
    yc=yc.Length;
    zc=nc_getdiminfo(snapnm,'k');
    zc=zc.Length;
    xarray=[xs:xs+xc-1];
    yarray=[ys:ys+yc-1];
    zarray=[zs:zs+zc-1];
    if (length(find(xarray>=gsubs(1)-1 & xarray<=gsube(1)-1)) ~= 0 && ...
        length(find(yarray>=gsubs(2)-1 & yarray<=gsube(2)-1)) ~= 0 && ...
        length(find(zarray>=gsubs(3)-1 & zarray<=gsube(3)-1)) ~= 0)
        n=n+1;

        px(n)=str2num(snaplist(i).name( strfind(snaplist(i).name,'px' )+2 : ...
                                         strfind(snaplist(i).name,'_py')-1));
        py(n)=str2num(snaplist(i).name( strfind(snaplist(i).name,'py' )+2 : ...
                                         strfind(snaplist(i).name,'_pz')-1));
        pz(n)=str2num(snaplist(i).name( strfind(snaplist(i).name,'pz' )+2 : ...
                                         strfind(snaplist(i).name,'.nc')-1));
    end
    
end

% retrieve the snapshot information
nthd=0;
for ip=1:length(px)
    
    nthd=nthd+1;
    
    snapnm=[snap_dir,'/',snapprefix,'_px',num2str(px(ip)),...
            '_py',num2str(py(ip)),'_pz',num2str(pz(ip)),'.nc'];
    xyzs=double(nc_attget(snapnm,nc_global,'first_index_to_snapshot_output'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xc=nc_getdiminfo(snapnm,'i');
    xc=xc.Length;
    yc=nc_getdiminfo(snapnm,'j');
    yc=yc.Length;
    zc=nc_getdiminfo(snapnm,'k');
    zc=zc.Length;
    xe=xs+xc-1;
    ye=ys+yc-1;
    ze=zs+zc-1;
    
    gxarray=gsubs(1):gsubt(1):gsube(1);
    gxarray=gxarray-1;
    gyarray=gsubs(2):gsubt(2):gsube(2);
    gyarray=gyarray-1;
    gzarray=gsubs(3):gsubt(3):gsube(3);
    gzarray=gzarray-1;
    
    snapinfo(nthd).thisid=[px(ip),py(ip),pz(ip)];
    i=find(gxarray>=xs & gxarray<=xe);
    j=find(gyarray>=ys & gyarray<=ye);
    k=find(gzarray>=zs & gzarray<=ze);
    snapinfo(nthd).indxs=[i(1),j(1),k(1)];
    snapinfo(nthd).indxe=[i(end),j(end),k(end)];
    snapinfo(nthd).indxc=snapinfo(nthd).indxe-snapinfo(nthd).indxs+1;
    
    snapinfo(nthd).subs=[ gxarray(i(1))-xs+1, ...
                          gyarray(j(1))-ys+1, ...
                          gzarray(k(1))-zs+1 ];
    snapinfo(nthd).subc=snapinfo(nthd).indxc;
    snapinfo(nthd).subt=gsubt;
    snapinfo(nthd).tinv=snap_tinv;
    snapinfo(nthd).fnmprefix=snapprefix;
end

end

