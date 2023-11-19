function metricinfo = locate_metric(parfnm,metric_dir,subs,subc,subt)

% check parameter file exist
if ~ exist(parfnm,'file')
    error([mfilename ': file ' parfnm ' does not exist']);
end

% read parameters file
par=loadjson(parfnm);
ngijk=[par.number_of_total_grid_points_x,...
       par.number_of_total_grid_points_y,...
       par.number_of_total_grid_points_z];

gsubs = subs;
gsubt = subt;
gsubc = subc;

% reset count=-1 to total number
indx=find(subc==-1);
gsubc(indx)=ceil((ngijk(indx)-gsubs(indx)+1)./gsubt(indx));

gsube=gsubs+(gsubc-1).*gsubt;

% search the nc file headers to locate the threads/processors
metricprefix='metric';
metriclist=dir([metric_dir,'/',metricprefix,'*.nc']);
n=0;
for i=1:length(metriclist)
    
    metricnm=[metric_dir,'/',metriclist(i).name];
    xyzs=double(nc_attget(metricnm,nc_global,'global_index_of_first_physical_points'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xyzc=double(nc_attget(metricnm,nc_global,'count_of_physical_points'));
    xc=xyzc(1);
    yc=xyzc(2);
    zc=xyzc(3);
    xarray=[xs:xs+xc-1];
    yarray=[ys:ys+yc-1];
    zarray=[zs:zs+zc-1];
   if (length(find(xarray>=gsubs(1)-1 & xarray<=gsube(1)-1)) ~= 0 && ...
       length(find(yarray>=gsubs(2)-1 & yarray<=gsube(2)-1)) ~= 0 && ...
       length(find(zarray>=gsubs(3)-1 & zarray<=gsube(3)-1)) ~= 0)
        n=n+1;

        px(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'px' )+2 : ...
                                         strfind(metriclist(i).name,'_py')-1));
        py(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'py' )+2 : ...
                                         strfind(metriclist(i).name,'_pz')-1));
        pz(n)=str2num(metriclist(i).name( strfind(metriclist(i).name,'pz' )+2 : ...
                                         strfind(metriclist(i).name,'.nc')-1));
    end

end

% retrieve the snapshot information
nthd=0;
for ip=1:length(px)
    
    nthd=nthd+1;
    
    metricnm=[metric_dir,'/',metricprefix,'_px',num2str(px(ip)),...
            '_py',num2str(py(ip)),'_pz',num2str(pz(ip)),'.nc'];
    xyzs=double(nc_attget(metricnm,nc_global,'global_index_of_first_physical_points'));
    xs=xyzs(1);
    ys=xyzs(2);
    zs=xyzs(3);
    xyzc=double(nc_attget(metricnm,nc_global,'count_of_physical_points'));
    xc=xyzc(1);
    yc=xyzc(2);
    zc=xyzc(3);
    xe=xs+xc-1;
    ye=ys+yc-1;
    ze=zs+zc-1;
    
    gxarray=gsubs(1):gsubt(1):gsube(1);
    gxarray=gxarray-1;
    gyarray=gsubs(2):gsubt(2):gsube(2);
    gyarray=gyarray-1;
    gzarray=gsubs(3):gsubt(3):gsube(3);
    gzarray=gzarray-1;
    
    metricinfo(nthd).thisid=[px(ip),py(ip),pz(ip)];
    i=find(gxarray>=xs & gxarray<=xe);
    j=find(gyarray>=ys & gyarray<=ye);
    k=find(gzarray>=zs & gzarray<=ze);
    metricinfo(nthd).indxs=[i(1),j(1),k(1)];
    metricinfo(nthd).indxe=[i(end),j(end),k(end)];
    metricinfo(nthd).indxc=metricinfo(nthd).indxe-metricinfo(nthd).indxs+1;
    
    metricinfo(nthd).subs=[ gxarray(i(1))-xs+1, ...
                            gyarray(j(1))-ys+1, ...
                            gzarray(k(1))-zs+1 ];
    metricinfo(nthd).subc=metricinfo(nthd).indxc;
    metricinfo(nthd).subt=gsubt;
    
    metricinfo(nthd).fnmprefix=metricprefix;
end

end

