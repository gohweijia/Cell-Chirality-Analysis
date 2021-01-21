

% Decompose the vectors into parallel & perpendicular components w.r.t
% centroid of sphere.
%
% Clockwise = negative, Counterclockwise = positive
%
% Written by hui ting, 22 Jan 2016.



function sign_theta = pp_pr_point(np_x2,np_y2,nc_x2,nc_y2,x_centroid,y_centroid)



%%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Find angle between vectors    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

costheta2 = nan(1,length(np_x2));  
for c_pl=1:length(np_x2) % for all x,y,u,v   
       
    
%   v_edge = [x_centroid-np_x2(c_pl) y_centroid-np_y2(c_pl)]; % vector pointing to circle centroid
    v_edge_pr = [-(y_centroid-np_y2(c_pl)) x_centroid-np_x2(c_pl)]; % v_edge_pr is perpendicular to v_edge ([a b] is perpendicular to [-b a])
  
    v_uv = [nc_x2(c_pl)-np_x2(c_pl) nc_y2(c_pl)-np_y2(c_pl)]; % vector pointing from ep near periphery to ep near circle
    
%   costheta(c_pl) = dot(v_edge,v_uv)/(norm(v_edge)*norm(v_uv)); % angle with radial vector  
    costheta2(c_pl) = dot(v_edge_pr,v_uv)/(norm(v_edge_pr)*norm(v_uv)); % angle with tangential vector

end


costheta2(isnan(costheta2))=0; 

theta_dg2 = acosd(costheta2); % 0 to 180 (angle with tangential vector) <90 = counterclockwise, >90 = clockwise



%%% ~~~~~~~~~~~~~~~~~~~~~~~    Classify vector to : clockwise and counterclockwise    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    idx_90_180_pr=find(theta_dg2>90);  % 90 to 180 (i.e, pr component with clockwise direction)
 
    theta_dg2(idx_90_180_pr)=180-theta_dg2(idx_90_180_pr);  % Make 0 to 180 become 0 to 90  



%%% ~~~~~~~~~~~~~~~      Angle to tangential axis, clockwise = positive, countercw = negative        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


sign_pr = ones(size(theta_dg2));         % counterclockwise = positive
sign_pr(idx_90_180_pr)=-1;               % clockwise = negative

sign_theta = sign_pr.*(90-theta_dg2);    % angle to radial axis with sign positive & negative






























