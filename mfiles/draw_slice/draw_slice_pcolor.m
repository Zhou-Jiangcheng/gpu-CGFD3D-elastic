clear all;
close all;
clc;
addmypath;
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/test.json';
output_dir='../../project/output';

% which slice to plot
slicedir='y';
sliceid=110;

% which variable and time to plot
varnm='Vx';
ns=501;
ne=501;
nt=100;

% figure control parameters
flag_km     = 1;
flag_print  = 0;
scl_daspect =[1 1 1];
clrmp       = 'jetwr';
taut=0.5;
% ---------------------------------------------------------------------- %

% read parameter file
par=loadjson(parfnm);

nproi=par.number_of_mpiprocs_x;
nproj=par.number_of_mpiprocs_y;
nprok=1;

% figure plot
hid=figure;
set(hid,'BackingStore','on');

% load data
for nlayer=ns:nt:ne
  % -------------------- slice x ---------------------- %
  if slicedir == 'x'
    [v, x, y, z, t] = gather_slice_x(output_dir,nlayer,varnm,sliceid,nproj,nprok);
        
    if flag_km
       x=x/1e3;
       y=y/1e3;
       z=z/1e3;
       str_unit='km';
    else
       str_unit='m';
    end

    pcolor(y,z,v);
    xlabel(['y axis (',str_unit,')']);
    ylabel(['z axis (',str_unit,')']);
  
  % -------------------- slice y ---------------------- %
  elseif slicedir == 'y'
    [v, x, y, z, t] = gather_slice_y(output_dir,nlayer,varnm,sliceid,nproi,nprok);
      
    if flag_km
       x=x/1e3;
       y=y/1e3;
       z=z/1e3;
       str_unit='km';
    else
       str_unit='m';
    end

      pcolor(x,z,v);
      xlabel(['x axis (',str_unit,')']);
      ylabel(['z axis (',str_unit,')']);
  
  % -------------------- slice z ---------------------- %
  else
    [v, x, y, z, t] = gather_slice_z(output_dir,nlayer,varnm,sliceid,nproi,nproj);
      
    if flag_km
       x=x/1e3;
       y=y/1e3;
       z=z/1e3;
       str_unit='km';
    else
       str_unit='m';
    end

    pcolor(x,y,v);
    xlabel(['x axis (',str_unit,')']);
    ylabel(['y axis (',str_unit,')']);
      
  end
  
  disp([ '  draw ' num2str(nlayer) 'th time step (t=' num2str(t) ')']);
  
  set(gca,'layer','top');
  set(gcf,'color','white','renderer','painters');

  % axis image
  % shading interp;
  shading flat;
  % colorbar range/scale
  if exist('scl_caxis')
      caxis(scl_caxis);
  end
  % axis daspect
  if exist('scl_daspect')
      daspect(scl_daspect);
  end
  % colormap and colorbar
  if exist('clrmp')
      colormap(clrmp);
  end
  colorbar('vert');
  
  % title
  titlestr=['Snapshot of ' varnm ' at ' ...
            '{\fontsize{12}{\bf ' ...
            num2str((t),'%7.3f') ...
            '}}s'];
  title(titlestr);
  
  drawnow;
  pause(taut);
  
  % save and print figure
  if flag_print==1
      width= 500;
      height=500;
      set(gcf,'paperpositionmode','manual');
      set(gcf,'paperunits','points');
      set(gcf,'papersize',[width,height]);
      set(gcf,'paperposition',[0,0,width,height]);
      fnm_out=[varnm '_ndim_',num2str(nlayer,'%5.5i')];
      print(gcf,[fnm_out '.png'],'-dpng');
  end
  
end
