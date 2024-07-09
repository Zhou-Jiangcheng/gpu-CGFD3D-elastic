clear all;
close all;
clc;
addmypath
% -------------------------- parameters input -------------------------- %
% file and path name
parfnm='../../project/test.json'
output_dir='../../project/output'

% which snapshot to plot
id=1;

%-- z slice
subs=[1,1,1];      % start from index '1'
subc=[-1,-1,1];     % '-1' to plot all points in this dimension
subt=[1,1,1];

%-- y slice
% subs=[1,11,1];      % start from index '1'
% subc=[-1,1,-1];     % '-1' to plot all points in this dimension
% subt=[1,1,1];

%-- x slice
% subs=[41,1,1];      % start from index '1'
% subc=[1,-1,-1];     % '-1' to plot all points in this dimension
% subt=[1,1,1];

% variable and time to plot
varnm='Vx';
ns=1;
ne=1001;
nt=100;


% figure control parameters
flag_km     = 0;
flag_print  = 0;
savegif = 0;
% scl_caxis=[-1.0 1.0];
filename1 = ['Vx.gif'];
scl_daspect =[1 1 1];
clrmp       = 'jetwr';
taut=0.5;
% ---------------------------------------------------------------------- %

% load snapshot data
snapinfo=locate_snap(parfnm,output_dir,id,subs,subc,subt);
% get coordinate data
coordinfo=locate_snap_coord(parfnm,output_dir,id,subs,subc,subt);
[x,y,z]=gather_coord(coordinfo,output_dir);

%- set coord unit
if flag_km
   x=x/1e3;
   y=y/1e3;
   z=z/1e3;
   str_unit='km';
else
   str_unit='m';
end

% figure plot
hid=figure;
set(hid,'BackingStore','on');

% snapshot show
for nlayer=ns:nt:ne
    
    [v,t]=gather_snap(snapinfo,nlayer,varnm,output_dir);
    
    disp([ '  draw ' num2str(nlayer) 'th time step (t=' num2str(t) ')']);
    
    if subc(1) == 1
        pcolor(y,z,v);
        xlabel(['Y axis (' str_unit ')']);
        ylabel(['Z axis (' str_unit ')']);
        
    elseif subc(2) == 1
        pcolor(x,z,v);
        xlabel(['X axis (' str_unit ')']);
        ylabel(['Z axis (' str_unit ')']);
        
    else
        pcolor(x,y,v);
        xlabel(['X axis (' str_unit ')']);
        ylabel(['Y axis (' str_unit ')']);
    end
    
    set(gca,'layer','top');
    set(gcf,'color','white','renderer','painters');

    % axis image
    % shading
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
    
    %title
    titlestr=['gpu Snapshot of ' varnm ' at ' ...
              '{\fontsize{12}{\bf ' ...
              num2str((t),'%7.3f') ...
              '}}s'];
    title(titlestr);
    
    drawnow;
    pause(taut);
    %save gif
    if savegif
      im=frame2im(getframe(gcf));
      [imind,map]=rgb2ind(im,256);
      if nlayer==ns
        imwrite(imind,map,filename1,'gif','LoopCount',Inf,'DelayTime',0.5);
      else
        imwrite(imind,map,filename1,'gif','WriteMode','append','DelayTime',0.5);
      end
    end
    
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


