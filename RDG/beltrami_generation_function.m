function [rou, tau] = beltrami_generation_function(r,t,k1,k2)
filter1=fspecial('average');
filter2=fspecial('gaussian',7,1000000);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BIG%%%%%%%%%%%%%%%%%%%%%%%%%%%
    k1=129;
    k2=129;
    [X, Y] = meshgrid(linspace(-10, 10, k1), linspace(-10, 10, k2));  
    peak_params1.x_center = rand*16-8;  
    peak_params1.y_center = rand*16-8;  
    peak_params1.sigma_x = rand*1.5+0.5; 
    peak_params1.sigma_y = rand*1.5+0.5; 
    peak_params1.amplitude = 0.99; 
    peak_params1.sign = randi([0, 1]) * 2 - 1; 
    
    peak_params2.x_center = rand*16-8;  
    peak_params2.y_center = rand*16-8; 
    peak_params2.sigma_x = rand*1.5+0.5; 
    peak_params2.sigma_y = rand*1.5+0.5; 
    peak_params2.amplitude = 0.99; 
    peak_params2.sign =  randi([0, 1]) * 2 - 1; 

    Z1 = sharp_peak_function(X, Y, peak_params1);
    Z2 = sharp_peak_function(X, Y, peak_params2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%SMALL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    rr=(2*rand(129,129)-1)/2;
    rr(1:5, :) = 0;
    rr(end-4:end, :) = 0;
    rr(:, 1:5) = 0;
    rr(:, end-4:end) = 0;
    rr=imfilter(rr,filter2);
    rr=imfilter(rr,filter2);
    
    tt=(2*rand(129,129)-1)/2;
    tt(1:5, :) = 0;
    tt(end-4:end, :) = 0;
    tt(:, 1:5) = 0;
    tt(:, end-4:end) = 0;
    tt=imfilter(tt,filter2);
    tt=imfilter(tt,filter2);
    figure;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rr=rr+Z1;   
    tt=tt+Z2;

    r = r + rr;
    t = t + tt;   

    logicalIndex1 = r.^2+t.^2 >= 1;  
    r(logicalIndex1) = r(logicalIndex1) ./ sqrt((r(logicalIndex1).^2 + t(logicalIndex1).^2 + 0.05));
    t(logicalIndex1) = t(logicalIndex1) ./ sqrt((r(logicalIndex1).^2 + t(logicalIndex1).^2 + 0.05));
    rou=r;
    tau=t;

% [x,y]=meshgrid(0:0.1:12.8,0:0.1:12.8);
% figure;
% quiver(x,y,r,t);
% axis equal
% % axis([0,128,0,128])
% axis off



% r=rou;
% t=tau;
% surf(r.^2+t.^2);
% rrrrr=r;
% ttttt=t;
% rowIndices = 1:5:size(r, 1);
% colIndices = 1:5:size(r, 2);
% 
% rrrr = r(rowIndices, colIndices);
% % surf(rrrr.^2+tttt.^2);
% rowIndices = 1:5:size(t, 1);
% colIndices = 1:5:size(t, 2);
% 
% tttt = t(rowIndices, colIndices);
% % surf(t);
% % surf(r);
% 
% [x,y]=meshgrid(0:1:25,0:1:25);
% figure;
% quiver(x,y,rrrr,tttt,0);
% axis equal
% % axis([0,128,0,128])
% axis off
% 
% exportgraphics(gcf, 'vector_field_scaled.png', 'Resolution', 3000);
% exportgraphics(gcf, 'vector_field_scaled.eps');

