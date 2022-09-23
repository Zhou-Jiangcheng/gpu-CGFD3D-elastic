clc;
clear all;
close all;
%x direction 21 stations
%y direction 22 stations
%all is 21*21 = 441
nx = 41;
ny = 41;
is_index = 0;
is_depth = 1;
depth = 0.0;
sta_num = nx*ny;
%origin is (0,0)
%space 1000
dh =1000;
%%
%==============================================================================
%-- write .gdlay file
%==============================================================================
station_file = 'station.list';

fid=fopen(station_file,'w'); % Output file name 

%-- first line: how many stations
fprintf(fid,'%6d\n',sta_num);

%-- second line: station name and coords
% on topography surface, z set 9999 
for i = 1:nx
    for j = 1:ny
        indx = (i-1)*ny + j-1;
        sta_name = ['recv',num2str(indx),'_x_y_z'];
        x = dh*(i-1);
        y = dh*(j-1);
        fprintf(fid,'%s %d %d %12.2f  %12.2f  %12.2f\n',sta_name, is_index, is_depth, x, y, depth);
    end
end
fclose(fid);
