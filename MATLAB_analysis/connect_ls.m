 

function [ls_cls_gap] = connect_ls(nc_x,nc_y,np_x,np_y,ls,no_ls,L_ep,cx,cy)

% For each near_centre_end_point, find any near_periphery_end_point that is near to it?

ls_cls_gap = ls;
L_cnn_rcd = [];
cnn_ang_chg_rcd = [];
cnn_cx_rcd = [];
cnn_cy_rcd = [];
cnn_px_rcd = [];
cnn_py_rcd = [];
z = 0;

title('ls');
imshow(ls,[]);hold on;
 
for ii=1:no_ls           
            
    dist_to_nc = sqrt((np_x-nc_x(ii)).^2+(np_y-nc_y(ii)).^2);
    dist_to_nc(ii) = nan; % remove itself
    i_sm_gap = find(dist_to_nc<=30); % 30    
    
    ang_chg = Inf.*ones(1,length(i_sm_gap));
    
    for jj = 1:length(i_sm_gap)
        
    y_df1 = np_y(ii)-nc_y(ii);
    x_df1 = np_x(ii)-nc_x(ii);
    
    y_df2 = np_y(i_sm_gap(jj))-nc_y(i_sm_gap(jj));
    x_df2 = np_x(i_sm_gap(jj))-nc_x(i_sm_gap(jj));
    
    y_dfg = nc_y(ii)-np_y(i_sm_gap(jj));
    x_dfg = nc_x(ii)-np_x(i_sm_gap(jj));       
    
    slp_ang1 = atan2d(y_df1,x_df1);
    slp_ang2 = atan2d(y_df2,x_df2);
    slp_angg = atan2d(y_dfg,x_dfg);
    
    % distance to centre point
    dist_to_centre1 = sqrt((nc_x(ii)-cx).^2+(nc_y(ii)-cy).^2);  % Point finding connection
    dist_to_centre2 = sqrt((np_x(i_sm_gap(jj))-cx).^2+(np_y(i_sm_gap(jj))-cy).^2); % Point to be connected
         
    ang_chg1 = abs(slp_ang1-slp_angg);
    ang_chg2 = abs(slp_ang2-slp_angg);
    if (ang_chg1<=30 && ang_chg2<=30) && (dist_to_centre1>dist_to_centre2)  % 30 
    ang_chg(jj) = ang_chg1+ang_chg2;        
    end
    
    end
    
    if sum(isinf(ang_chg))~=length(ang_chg) % if there is any candidate for connection, choose the one with minimum angle change
        
    [cnn_ang_chg,jj_min] = min(ang_chg);
    cnn_px = np_x(i_sm_gap(jj_min));
    cnn_py = np_y(i_sm_gap(jj_min));
     
    L_cnn = L_ep(cnn_py,cnn_px);
    
    if ~ismember(L_cnn,L_cnn_rcd) % if candidate not connected before       
    
        z = z+1;
        L_cnn_rcd(z) = L_cnn;
        cnn_ang_chg_rcd(z) = cnn_ang_chg;
        cnn_cx_rcd(z) = nc_x(ii);
        cnn_cy_rcd(z) = nc_y(ii);
        cnn_px_rcd(z) = cnn_px;
        cnn_py_rcd(z) = cnn_py;    

    
    else % if candidate connected before, compare angle change
        
        q = find(L_cnn_rcd == L_cnn);
        if cnn_ang_chg < cnn_ang_chg_rcd(q) % Best match, update
                        
            cnn_ang_chg_rcd(q) = cnn_ang_chg;
            cnn_cx_rcd(q) = nc_x(ii);
            cnn_cy_rcd(q) = nc_y(ii);
            cnn_px_rcd(q) = cnn_px;
            cnn_py_rcd(q) = cnn_py;
            
        end
         
    end
        
    
    
    end
end
   
        % Finish compare & decide best match
        for qq = 1:length(L_cnn_rcd)   
        hLine = imline(gca,[cnn_cx_rcd(qq) cnn_px_rcd(qq)],[cnn_cy_rcd(qq) cnn_py_rcd(qq)]);
        ls_cls_gap_temp = hLine.createMask();  % create line mask for gap and add it to line segments mask
        ls_cls_gap = ls_cls_gap+ls_cls_gap_temp;  
        end


close all;

end







