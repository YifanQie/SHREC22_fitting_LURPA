close all;
clear;
clc;

% %%%%
% 1. read the data from .txt file, find the vertex
% 2. pre-process the point cloud
% 3. save as figure
% %%%%

tic % strat counting the time
%%%%%%%%%%%%%%%%% 1. read the data from .txt file with the vertex for processing %%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%for all the surfaces %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% read the point clouds from the folder %%%%%
namelist = dir('D:\matlab workplace\partition\Measurement test\SHREC22 for view\*.txt');
len = length(namelist); % two folders same size
% natural order
[~, index] = natsort({namelist.name});
namelist_order = namelist(index);

%%%% read the labels from the folder %%%%%
labellist = dir('D:\matlab workplace\partition\Measurement test\SHREC22 labels\*.txt');
% natural order
[~, index_label] = natsort({labellist.name});
labellist_order = labellist(index_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% group processing the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1: len % = 103
    % offical code start here%
    filenname = labellist_order(i).name
    surface_type = importdata(filenname); % surface_type(1) the first number is the surface type

    % load the points in one file
    filenname = namelist_order(i).name;
    L = importdata(filenname);
    process_point_cloud(L,i,surface_type(1));
    % process_point_cloud(L,i,1);
   
end


toc


function process_point_cloud(Point_cloud,i, surface_type)

    % move the point cloud to the origin and scale it in a unit sphere
    % 1. find the PCA of the point cloud and rotate it along the x-y-z axis
    Point_cloud = rotatePointCloudAlongZ(Point_cloud, 'x');

    %2 zoom the poitn into a unit shpere
    %2.1 find the largest abs value
    dist = max(sqrt(sum(Point_cloud.*Point_cloud,2)));
    Point_cloud = Point_cloud./dist; 
    indexcolor = Point_cloud(:,2);
    indexcolor(abs(indexcolor) < 10^(-3)) = 0;

    % prepare the figure for training
    h_fig = figure('Visible', 'of');      
    scatter3(Point_cloud(:,1), Point_cloud(:,2), Point_cloud(:,3), 50, indexcolor,'filled');
    axis equal
    axis off
    view(0,0)
    % caxis([-0.5 0.5])
    % colormap(parula(20))
    % colormap('hsv') 

    filename = strcat('traingset_',num2str(i));
    filename = strcat(filename,'.png');
   
    saveas(gcf,['D:\matlab workplace\partition\Measurement test\DL_data_trainingset\' num2str(surface_type) '\traingset_', num2str(i) '.png']); 
    close(h_fig);

end