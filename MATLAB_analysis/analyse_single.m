function analyse_single(pathname, from_, to_, nb_thresh, min_length_um, pixel_size)

    dist_fr_edge_um = to_;
    cell_edge_init = round(from_/pixel_size)
    min_length_pixels = round(min_length_um/pixel_size);
    dist_fr_edge_pxl = round(dist_fr_edge_um/pixel_size)
    nb_offset = -0.01;                            
    scale_mean_int = 1;
    gss = fspecial('gaussian',[15 15],1.5);       % 2,3
    rolling_ball = strel('ball',30,30);   % background subtraction
    clr ={'r','g','y','c','m'};

    output_file = fullfile(pathname,'actin',[num2str(from_), '_', num2str(to_)], 'Results.csv');
    pathname_cell = fullfile(pathname,'actin');
    pathname_fiber = fullfile(pathname,'actin','Masks','Unet_Resnet50-20191017T0347_0155')

    cell_img_files = dir(fullfile(pathname_cell, '*.tif'));
    fiber_img_files = dir(fullfile(pathname_fiber,'*.tif'));
    total_files = length(fiber_img_files)
    output_folder = fullfile(pathname,'actin',[num2str(from_), '_', num2str(to_)]);
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
    disp(pathname_cell);
    disp(cell_img_files(1).name);
    cell_img_temp = imread(fullfile(pathname_cell,cell_img_files(1).name),1);
    [r,c] = size(cell_img_temp);
    all_bw = zeros(r,c,total_files);

    for ifile = 1:total_files

        cell_filename = cell_img_files(ifile).name
        fiber_filename = fiber_img_files(ifile).name

        cell_img = imread(fullfile(pathname_cell,cell_filename),1);
        fiber_img = imread(fullfile(pathname_fiber,fiber_filename),1);

        sm_img = double(imfilter(cell_img, gss,'replicate'));    % figure;imshow(sm_green_data,[]);

        [bw_circle, cx, cy] = get_circle_centre(sm_img);         % Get circle centre.
        all_bw(:,:,ifile) = bw_circle;

        dist_map = bwdist(~bw_circle);
        bw_roi = dist_map<=dist_fr_edge_pxl&dist_map>cell_edge_init;

        [x,cellname,x] = fileparts(cell_filename);
        mask_name = sprintf('%s_mask.tif', cellname);
        mask_path = char(fullfile(output_mask_folder, mask_name));
        
        roi_name = sprintf('%s_roi.tif', cellname);
        roi_path = char(fullfile(output_roi_folder, roi_name));
        
        imwrite(bw_circle, mask_path,'compression','none');
        imwrite(bw_roi, roi_path,'compression','none');
    

        
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


        sk = bwmorph(ls_cls_gap,'skel');
        bp = bwmorph(sk,'branchpoints');
        dbp = imdilate(bp,strel('disk',1));  % 3                                                           % figure;imshow(dbp,[]);

        rmv_bp = sk-dbp;
        rmv_bp = rmv_bp>0;
        rmv_bp = bwmorph(rmv_bp, 'spur', 3);  % added line 05Feb-- remove pixels from end of RFs


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

            cell_area = sum(bw_circle(:));
      

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            for iii=1:total_fiber

                [yl,xl] = find(L_fiber==iii);

                clr_idx = mod(iii,5)+1;
                scatter(xl,yl,1,clr{clr_idx},'.');
                ht = text(xl(1),yl(1),num2str(iii));ht.Color = clr{clr_idx};ht.FontSize = 7;hold on;
            end

            fiber_ID{ifile} = [1:total_fiber]';
            fiber_ornt{ifile} = sign_theta';
            fiber_edgex{ifile} = np_x2';
            fiber_edgey{ifile} = np_y2';
            cell_x{ifile} = repmat(cx',[total_fiber 1]);
            cell_y{ifile} = repmat(cy',[total_fiber 1]);
            cell_area_temp{ifile} = repmat(cell_area',[total_fiber 1]);    
        else
            fiber_ID{ifile} = NaN([total_fiber 1]);
            fiber_ornt{ifile} = NaN([total_fiber 1]);
            fiber_edgex{ifile} = NaN([total_fiber 1]);
            fiber_edgey{ifile} = NaN([total_fiber 1]);
            cell_x{ifile} = NaN([total_fiber 1]);
            cell_y{ifile} = NaN([total_fiber 1]);
            cell_area_temp{ifile} = NaN([total_fiber 1]);   
        end

        fg1 = getframe;cla;
    
        fiberout_name = sprintf('%s_fiber ID.tif', cellname);
        fiberout_path = char(fullfile(output_fiber_folder, fiberout_name));
        imwrite(fg1.cdata, fiberout_path);

        filename_out = repmat(fiber_img_files(ifile).name,[total_fiber 1]);
        count_all = count_all+total_fiber;
        total_fiber_out(ifile) = total_fiber;
        avg_fiber_ornt(ifile) = nanmean(fiber_ornt{ifile});
        frame_temp{ifile} = repmat(ifile,[total_fiber 1]);

        close all;

    end

    fiber_ID_out = cell2mat(fiber_ID');
    fiber_ornt_out = cell2mat(fiber_ornt');
    fiber_edgex_out = cell2mat(fiber_edgex');
    fiber_edgey_out = cell2mat(fiber_edgey');
    cell_area_out = cell2mat(cell_area_temp');
    cellx_out = cell2mat(cell_x');
    celly_out = cell2mat(cell_y');


    max_all_bw = max(all_bw,[],3);  % 512, 512
    min_all_bw = min(all_bw,[],3);

    frame_out = cell2mat(frame_temp');  
    frame_out = frame_out(~isnan(fiber_ID_out));
    fiber_ID_out = fiber_ID_out(~isnan(fiber_ID_out));  %
    fiber_ornt_out = fiber_ornt_out(~isnan(fiber_ornt_out));  %
    file_index = str2double((fiber_img_files(1).name(1:4)));
    cellx_out = cellx_out(~isnan(cellx_out));
    celly_out = celly_out(~isnan(celly_out));
    cell_area_out =  cell_area_out(~isnan(cell_area_out));

    output_table = table(frame_out, fiber_ID_out, fiber_ornt_out, fiber_edgex_out, fiber_edgey_out, cell_area_out, cellx_out, celly_out);
    output_table.Properties.VariableNames = {'File_index', 'Fiber_ID', 'Fiber_Orientation', 'Edge_x', 'Edge_y', 'Cell_area', 'Cell_x', 'Cell_y'};
    writetable(output_table, fullfile(output_folder, 'Results.csv'));
end

