


function [pixel_size,nb_thresh,min_length_pixels,dist_fr_edge_pxl,dist_fr_edge_um] = get_user_input


pixel_size = 0.138502;            % micrometer
nb_thresh = -0.3;
min_length_um = 1;         
dist_fr_edge_um = 5;           


prompt={'Pixel size (\mum/pixel)','Niblack local threshold (-0.5 to 0.5)','Minimum fiber length (\mum)','Distance from cell edge for ROI (\mum)'};
name='Input';     numlines=1;
defaultanswer={num2str(pixel_size),num2str(nb_thresh),num2str(min_length_um),num2str(dist_fr_edge_um)};     
options.Resize='on';    options.WindowStyle='normal';    options.Interpreter='tex';
answer=inputdlg(prompt,name,numlines,defaultanswer,options);


pixel_size = str2double(answer{1})
nb_thresh = str2double(answer{2})  
min_length_um = str2double(answer{3})       
dist_fr_edge_um = str2double(answer{4})       


dist_fr_edge_pxl = round(dist_fr_edge_um/pixel_size)
min_length_pixels = round(min_length_um/pixel_size)











