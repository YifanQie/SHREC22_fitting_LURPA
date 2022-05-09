close all;
clear;
clc;

tic
% %%%%
% 1. read the data from .txt file, find the vertex
% 2. calculate the curvatures and show the value of the curvatures on the file
% 3. region growing by curvedness
% 4. Local shape type identitfication
% 5. visualization of the surface types
% %%%%

tic % strat counting the time
%%%%%%%%%%%%%%%%% 1. read the data from .obj file, find the vertex for processing %%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%for all the parts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% read the point clouds from the folder %%%%%

namelist = dir('D:\matlab workplace\partition\Measurement test\SHREC22 for view\*.txt');
len = length(namelist);
% natural order
[~, index] = natsort({namelist.name});
namelist_order = namelist(index);

%
% group processing the data
% figure;
count = 1;

for i = count: len %len % = 103

    % offical code start here%
    % load the points in one file
    L = importdata(namelist_order(i).name);

    % figure
    % scatter3(L(:,1), L(:,2),L(:,3),10,L(:,3) );
    % axis on
    % axis equal

    data_image = process_point_cloud(L,i);
   
end


toc


function data_image = process_point_cloud(Point_cloud,i)

    % move the point cloud to the origin and scale it in a unit sphere
    % 1. find the PCA of the point cloud and rotate it along the x-y-z axis
    direction = 'x';
    Point_cloud = rotatePointCloudAlongZ(Point_cloud, direction);

    %2 zoom the poitn into a unit shpere
    %2.1 find the largest abs value
    dist = max(sqrt(sum(Point_cloud.*Point_cloud,2)));
    Point_cloud = Point_cloud./dist;
    figure;
    scatter3(Point_cloud(:,1), Point_cloud(:,2), Point_cloud(:,3), 50, Point_cloud(:,2),'filled');
    axis equal
    axis off
    view(0,0)
    % caxis([-1 1])
    % colormap(parula(15))
    filename = strcat('test_',num2str(i));
    filename = strcat(filename,'.png');
    saveas(gcf, filename);

    data_image = 0;
end