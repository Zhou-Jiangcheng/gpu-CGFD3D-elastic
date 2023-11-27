clear all;
close all;
clc;
addmypath
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/test.json';
output_dir='../../project/output';

% which slice to plot
% slice 1
slicedir{1}='y';
sliceid{1}=100;
% slice 2
slicedir{2}='x';
sliceid{2}=100;
% slice 3
slicedir{3}='z';
sliceid{3}=100;

% which variable and time to plot
varnm='Vz';
ns=100;
ne=800;
nt=50;

% figure control parameters
flag_km     = 1;
flag_print  = 0;
scl_daspect =[1 1 1];
clrmp       = 'parula';
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
  for i=1:length(sliceid)
    % -------------------- slice x ---------------------- %
    if slicedir{i} == 'x'

    [v{i}, x{i}, y{i}, z{i}, t] = gather_slice_x(output_dir,nlayer,varnm,sliceid{i},nproj,nprok);

    if flag_km
       x{i}=x{i}/1e3;
       y{i}=y{i}/1e3;
       z{i}=z{i}/1e3;
       str_unit='km';
    else
       str_unit='m';
    end
    
    surf(x{i},y{i},z{i},v{i});
    xlabel(['x axis (',str_unit,')']);
    ylabel(['y axis (',str_unit,')']);
    zlabel(['z axis (',str_unit,')']);
    view(45,15);
     
    % -------------------- slice y ---------------------- %
    elseif slicedir{i} == 'y'
          
    [v{i}, x{i}, y{i}, z{i}, t] = gather_slice_y(output_dir,nlayer,varnm,sliceid{i},nproi,nprok);
    
    if flag_km
       x{i}=x{i}/1e3;
       y{i}=y{i}/1e3;
       z{i}=z{i}/1e3;
       str_unit='km';
    else
       str_unit='m';
    end

    surf(x{i},y{i},z{i},v{i});
    xlabel(['x axis (',str_unit,')']);
    ylabel(['y axis (',str_unit,')']);
    zlabel(['z axis (',str_unit,')']);
    view(45,15);
         
    % -------------------- slice z ---------------------- %
    else

    [v{i}, x{i}, y{i}, z{i}, t] = gather_slice_z(output_dir,nlayer,varnm,sliceid{i},nproi,nproj);
            
    if flag_km
       x{i}=x{i}/1e3;
       y{i}=y{i}/1e3;
       z{i}=z{i}/1e3;
       str_unit='km';
    else
       str_unit='m';
    end

    
    surf(x{i},y{i},z{i},v{i});
    xlabel(['x axis (',str_unit,')']);
    ylabel(['y axis (',str_unit,')']);
    zlabel(['z axis (',str_unit,')']);
    
    end
    
    hold on;
  end
  hold off;
    
  disp([ '  draw ' num2str(nlayer) 'th time step (t=' num2str(t) ')']);
  
  set(gca,'layer','top');
%   set(gcf,'color','white','renderer','painters');

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

        
        
        
