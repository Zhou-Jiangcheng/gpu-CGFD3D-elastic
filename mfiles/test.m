close all;
figure(1)
plot(seismot(1,:),seismodata(1,:),'b','linewidth',1.0);
hold on;
plot(seismot(1,:),data1(1,:),'r','linewidth',1.0);

xlabel('Time (s)');
ylabel('Amplitude');
title([varnm, ' recv No.',num2str(1),' interpreter ','yes']);
set(gcf,'color','white','renderer','painters');
