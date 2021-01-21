

function [nc_x,nc_y,np_x,np_y] = class_ep(L_ep,no_ls,cx,cy)


nc_x = nan(1,no_ls);
nc_y = nc_x;
np_x = nc_x; 
np_y = nc_x;


for i=1:no_ls
    
    [y,x] = find(L_ep==i);
    
    dist_to_centre = sqrt((x-cx).^2+(y-cy).^2);
    
    if length(dist_to_centre)==2
    
    [~,idx_nc] = min(dist_to_centre);
    
    idx_np = setdiff([1 2],idx_nc);
    
    nc_x(i) = x(idx_nc); % near circle centre end point
    nc_y(i) = y(idx_nc);
    np_x(i) = x(idx_np); % near circle periphery end point
    np_y(i) = y(idx_np);
           
    % plot(np_x(i),np_y(i),'r+');hold on;plot(nc_x(i),nc_y(i),'y+');     
    
    else
        
    nc_x(i) = nan; % near circle centre end point
    nc_y(i) = nan;
    np_x(i) = nan; % near circle periphery end point
    np_y(i) = nan;
    
    end

    
end