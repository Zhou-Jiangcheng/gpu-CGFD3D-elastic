function [v] = gather_metric(metricinfo,varnm,metric_dir)

% check path exists
if ~ exist(metric_dir,'dir')
   error([mfilename ': directory ' metric_dir ' does not exist']);
end

% load
metricprefix='metric';
nthd=length(metricinfo);
for n=1:nthd
    
    n_i=metricinfo(n).thisid(1); n_j=metricinfo(n).thisid(2);n_k=metricinfo(n).thisid(3);
    i1=metricinfo(n).indxs(1); j1=metricinfo(n).indxs(2); k1=metricinfo(n).indxs(3);
    i2=metricinfo(n).indxe(1); j2=metricinfo(n).indxe(2); k2=metricinfo(n).indxe(3);
    subs=metricinfo(n).subs; subc=metricinfo(n).subc; subt=metricinfo(n).subt;
    fnm_metric=[metric_dir,'/',metricprefix,'_px',num2str(n_i),'_py',num2str(n_j),'_pz',num2str(n_k),'.nc'];
    
    if ~ exist(fnm_metric,'file')
       error([mfilename ': file ' fnm_metric 'does not exist']);
    end

    subs = fliplr(subs); subc = fliplr(subc); subt = fliplr(subt);

    v(k1:k2,j1:j2,i1:i2)=nc_varget(fnm_metric,varnm, subs-1,subc,subt);
                               
end

v=squeeze(v);

end
