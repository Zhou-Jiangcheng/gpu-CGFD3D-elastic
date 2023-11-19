clear all;
close all;
clc;
addmypath
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/test.json';
output_dir='../../project/output';

% metric profiles to plot
% profile 1
subs{1}=[50,1,1];      % start from index '1'
subc{1}=[1,-1,-1];     % '-1' to plot all points in this dimension
subt{1}=[1,1,1];
% profile 2
subs{2}=[1,1,30];      % start from index '1'
subc{2}=[-1,-1,1];     % '-1' to plot all points in this dimension
subt{2}=[1,1,1];
% profile 3
subs{3}=[1,50,1];      % start from index '1'
subc{3}=[-1,1,-1];     % '-1' to plot all points in this dimension
subt{3}=[1,1,1];

% variable to plot
% 'jac', 'xi_x', 'xi_y', 'xi_z', 'eta_x', 'eta_y', 'eta_z',
% 'zeta_x', 'zeta_y', 'zeta_z'
varnm='jac';

% figure control parameters
flag_km     = 1;
flag_emlast = 1;
flag_print  = 0;
flag_clb    = 1;
flag_title  = 1;
scl_daspect = [1 1 1];
clrmp       = 'parula';
% ---------------------------------------------------------------------- %

% figure plot
hid=figure;
set(hid,'BackingStore','on');

for i=1:length(subs)
    
    % locate metric
    metricinfo{i}=locate_metric(parfnm,output_dir,subs{i},subc{i},subt{i});
    % get coordinate data
    [x{i},y{i},z{i}]=gather_coord(metricinfo{i},output_dir);
    %- set coord unit
    if flag_km
       x{i}=x{i}/1e3;
       y{i}=y{i}/1e3;
       z{i}=z{i}/1e3;
       str_unit='km';
    else
       str_unit='m';
    end
    
    % gather metric data
    v{i}=gather_metric(metricinfo{i},varnm,output_dir);
    
    % metric show
    surf(x{i},y{i},z{i},v{i});
    hold on;
    
end

xlabel(['X axis (' str_unit ')']);
ylabel(['Y axis (' str_unit ')']);
zlabel(['Z axis (' str_unit ')']);

set(gca,'layer','top');
set(gcf,'color','white','renderer','painters');

% shading
% shading interp;
shading flat;
% colorbar range/scale
if exist('scl_caxis','var')
    caxis(scl_caxis);
end
% axis daspect
if exist('scl_daspect')
    daspect(scl_daspect);
end
axis tight
% colormap and colorbar
if exist('clrmp')
    colormap(clrmp);
end
if flag_clb
    cid=colorbar;
end

% title
if flag_title
    title(varnm,'interpreter','none');
end

% save and print figure
if flag_print
    width= 500;
    height=500;
    set(gcf,'paperpositionmode','manual');
    set(gcf,'paperunits','points');
    set(gcf,'papersize',[width,height]);
    set(gcf,'paperposition',[0,0,width,height]);
    print(gcf,[varnm '.png'],'-dpng');
end


