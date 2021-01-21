


function [bw_circle, cx, cy] = get_circle_centre(green)

scale_otsu = 0.4;

gss = fspecial('gaussian',[35 35],3);
 
sm_green = imfilter(green,gss,'replicate');                                            % figure;imshow(sm_green,[]);

sm01 = mat2gray(sm_green);
lvl = graythresh(sm01);
bw = im2bw(sm01, scale_otsu*lvl);  
bw = imclose(bw,strel('disk',7));
bw_circle = imfill(bw,'holes');  
bw_circle = imerode(bw_circle,strel('disk',2));
L_circle = bwlabel(bw_circle);
area = regionprops(L_circle,'Area');
area = [area.Area];
bw_circle = ismember(L_circle,find(area==max(area)));
[r,c]=find(bw_circle);

cx = mean(mean(c));
cy = mean(mean(r));

% figure;imshow(bw_circle,[]);hold on;plot(cx,cy,'r+');
