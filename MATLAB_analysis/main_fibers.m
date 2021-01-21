

%  MAKE SURE THE FILENAMES OF CELLS AND FIBERS ARE MATCHED !!!!!!!!!!!!!!
%
%  This code segments radial fibers. 
%   
%  The input folder should be the main folder consists of data folders. 
%
%  Written by hui ting, 23 Sep 2019



close all;clear all;clc;

% ap;


%%%%%%%%%%%%%%%%%%%%%%%%       Input & initialization       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


pathname = uigetdir(cd, 'Select main input folder');  % main folder

[pixel_size,nb_thresh,min_length_pixels,dist_fr_edge_pxl,dist_fr_edge_um] = get_user_input;

data_folder = dir(pathname);
dir_yes = [data_folder.isdir] & ~strcmp({data_folder.name},'.') & ~strcmp({data_folder.name},'..') & ~strcmp({data_folder.name},'plots');
data_folder = data_folder(dir_yes);
total_data_folder = length(data_folder)

nb_offset = -0.01;                            % 0   (For fiber segmentation), increase offset will detect more, decrease offset will detect less
scale_mean_int = 1;
gss = fspecial('gaussian',[15 15],1.5);       % 2,3
rolling_ball = strel('ball',30,30);   % background subtraction
clr ={'r','g','y','c','m'};


%%%%%%%%%%%%%%%%%%%%%%%       For each data folder        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for c_df = 1:total_data_folder

pathname_cell = fullfile(pathname,data_folder(c_df).name,'actin');
pathname_fiber1 = dir(fullfile(pathname,data_folder(c_df).name,'actin','masks'));
dir_yes2 = [pathname_fiber1.isdir] & ~strcmp({pathname_fiber1.name},'.') & ~strcmp({pathname_fiber1.name},'..'); 
fiber_folder = pathname_fiber1(dir_yes2);
pathname_fiber = fullfile(pathname,data_folder(c_df).name,'actin','masks',fiber_folder(1).name);

cell_img_files = dir(fullfile(pathname_cell,'*.tif'));
fiber_img_files = dir(fullfile(pathname_fiber,'*.tif'));

total_files = length(fiber_img_files)

output_folder = fullfile(pathname,data_folder(c_df).name,['Results_' num2str(dist_fr_edge_um) 'um from edge']);
output_mask_folder = fullfile(output_folder,'mask');
output_roi_folder = fullfile(output_folder,'ROI');
output_fiber_folder = fullfile(output_folder,'fiber ID');
mkdir(output_mask_folder);
mkdir(output_roi_folder);
mkdir(output_fiber_folder);

total_fiber_out = nan(total_files,1);
avg_fiber_ornt = total_fiber_out;
count_all = 0;
clear fiber_ID fiber_ornt 

cell_img_temp = imread(fullfile(pathname_cell,cell_img_files(1).name),1); 
[r,c] = size(cell_img_temp);
all_bw = zeros(r,c,total_files);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       For each file        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for ifile = 1:total_files   

cell_filename = cell_img_files(ifile).name
fiber_filename = fiber_img_files(ifile).name

cell_img = imread(fullfile(pathname_cell,cell_filename),1); 
fiber_img = imread(fullfile(pathname_fiber,fiber_filename),1); 

sm_img = double(imfilter(cell_img, gss,'replicate'));    % figure;imshow(sm_green_data,[]);

[bw_circle, cx, cy] = get_circle_centre(sm_img);         % Get circle centre. 
all_bw(:,:,ifile) = bw_circle;

dist_map = bwdist(~bw_circle);
bw_roi = dist_map<=dist_fr_edge_pxl&dist_map>0;

imwrite(bw_circle,fullfile(output_mask_folder,[cell_filename '_mask.tif']),'compression','none');
imwrite(bw_roi,fullfile(output_roi_folder,[cell_filename '_roi.tif']),'compression','none');

                                                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       Segmentation        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


bs_fiber_img = imtophat(fiber_img,rolling_ball);
bs_fiber_img = double(bw_roi).*double(bs_fiber_img); % masked with ROI
[ls, L_ls, total_ls] = seg_radial_fiber(bs_fiber_img,bw_circle,nb_thresh,nb_offset,scale_mean_int);  % Get line segments of radial fiber 


%%%%%%%%%%%%%%%%%%%%%%%%%%       Connect line segments to get fiber       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


all_ep = bwmorph(ls,'endpoints');                                                       % figure;imshow(ls,[]);hold on;
L_ep = all_ep.*L_ls;                                                                    % figure;imshow(L_ep,[]);colormap jet;hold on;

[nc_x,nc_y,np_x,np_y] = class_ep(L_ep,total_ls,cx,cy);  % Classify each pair of end points to the end_point_near_circle_centre or end_point_near_periphery
  
[ls_cls_gap] = connect_ls(nc_x,nc_y,np_x,np_y,ls,total_ls,L_ep,cx,cy);                   % figure;imshow(ls,[]);hold on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Remove branchpoints        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sk = bwmorph(ls_cls_gap,'thin',Inf);                                                                   % figure;imshow(sk,[]);
bp = bwmorph(sk,'branchpoints');  
dbp = imdilate(bp,strel('disk',1));  % 3                                                           % figure;imshow(dbp,[]);

rmv_bp = sk-dbp; 
rmv_bp = rmv_bp>0;

L_rmv_bp = bwlabel(rmv_bp); 
aaa = regionprops(L_rmv_bp,'Area');	
aaa =[aaa.Area];
bw_fiber = ismember(L_rmv_bp,find(aaa>=min_length_pixels)); % 5
L_fiber = bwlabel(bw_fiber);                                                                     % imshow(L_fiber,[]);colormap jet;hold on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       Orientation       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


all_ep2 = bwmorph(bw_fiber,'endpoints'); %%% classify end points                                                      
L_ep2 = all_ep2.*L_fiber;   
total_fiber = max(L_fiber(:));

imshow(bs_fiber_img,[]);hold on;

if total_fiber>0

[nc_x2,nc_y2,np_x2,np_y2] = class_ep(L_ep2,total_fiber,cx,cy);  % Classify each pair of end points to the end_point_near_circle_centre or end_point_near_periphery

sign_theta = pp_pr_point(np_x2,np_y2,nc_x2,nc_y2,cx,cy); % Angle with tangential axis, positive = counterclockwise, negative = clockwise


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                                                                                                              
for iii=1:total_fiber
    
    [yl,xl] = find(L_fiber==iii);      
          
    clr_idx = mod(iii,5)+1;
    scatter(xl,yl,1,clr{clr_idx},'.');
    ht = text(xl(1),yl(1),num2str(iii));ht.Color = clr{clr_idx};ht.FontSize = 7;hold on;
          
end

fiber_ID{ifile} = [1:total_fiber]';
fiber_ornt{ifile} = sign_theta';


else
    
fiber_ID{ifile} = nan;
fiber_ornt{ifile} = nan;
    
end
    
fg1 = getframe;cla;
imwrite(fg1.cdata,fullfile(output_fiber_folder,[fiber_filename '_fiber ID.tif']));


filename_out = repmat(fiber_img_files(ifile).name,[total_fiber 1]);
xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),cellstr(filename_out),'2',['A' num2str(count_all+2)]);
count_all = count_all+total_fiber;

total_fiber_out(ifile) = total_fiber;
avg_fiber_ornt(ifile) = nanmean(fiber_ornt{ifile});
frame_temp{ifile} = repmat(ifile,[total_fiber 1]);

close all;

end
 
fiber_ID_out = cell2mat(fiber_ID');
fiber_ornt_out = cell2mat(fiber_ornt');
frame_out = cell2mat(frame_temp');

fiber_ID_out = fiber_ID_out(~isnan(fiber_ID_out));
fiber_ornt_out = fiber_ornt_out(~isnan(fiber_ornt_out));

max_all_bw = max(all_bw,[],3);
min_all_bw = min(all_bw,[],3);

area_max = sum(max_all_bw(:));
area_min = sum(min_all_bw(:));

area_percentage = (area_min/area_max)*100;

xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),{'Filename','Total number of fibers','Average fiber orientation','Cell area percentage'},'1','A1');
xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),{fiber_img_files(:).name}','1','A2');
xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),[total_fiber_out avg_fiber_ornt],'1','B2');
xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),area_percentage,'1','D2');

xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),{'Filename','Frame','Fiber ID','Fiber orientation'},'2','A1');
xlswrite(fullfile(output_folder,[fiber_img_files(1).name 'Results.xlsx']),[frame_out fiber_ID_out fiber_ornt_out],'2','B2');





end





 
 
 
 
 
 
 









