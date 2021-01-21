

function [ls, L_ls, total_ls] = seg_radial_fiber(sm_green,bw_circle,nb_thresh,nb_offset,scale_mean_int)

mean_sm_green = mean(sm_green(bw_circle>0)); % intensity inside circle
% mean_sm_green = mean(sm_green(:));                                                              % figure;imshow(sm_green,[]);

% nb_thresh = -0.3; win = [15 15]
bw = niblack(sm_green, [15 15], nb_thresh, nb_offset, 'replicate');                                    % figure;imshow(bw,[]);
%bw = niblack(sm_green, [25 25], 0.4, nb_offset, 'replicate');                                    figure;imshow(bw,[]);

L=bwlabel(bw);
int = regionprops(L,sm_green,'MeanIntensity');
a = regionprops(L,'Area');

bw2 = ismember(L,find([int.MeanIntensity]>=scale_mean_int*mean_sm_green&[a.Area]>=7)); % 50    % figure;imshow(bw2,[]);

sk = bwmorph(bw2,'thin',Inf);                                                                   % figure;imshow(sk,[]);
bp = bwmorph(sk,'branchpoints');  
dbp = imdilate(bp,strel('disk',3)); % 3                                                         % figure;imshow(dbp,[]);

rmv_bp = sk-dbp; 
rmv_bp = rmv_bp>0;


rd_sk = bwmorph(rmv_bp,'thin',Inf);                                                                  % figure;imshow(rd_sk,[]);hold on;
ls = bwareaopen(rd_sk,3); % 6   

L_ls = bwlabel(ls);                                                                              % figure;imshow(L_ls,[]);title('out_ls');
total_ls = max(L_ls(:));





