close all;
clear;
clc;

% %%%%
% 1 load the NN
% 2 read the data from point cloud file
% 3 detect surface type
% 4 fiiting by the corresponding surface type
% %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1 load the NN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load AlexNet_v1_test;
% load my_net_trained;
load my_net_trained_opti;
net  = netTransfer; % for identifying surface type

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2. read the data from point cloud file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% read the point clouds from the folder %%%%%
namelist = dir('D:\matlab workplace\partition\Measurement test\pointCloud\*.txt');
len = length(namelist); % two folders same size
% natural order
[~, index] = natsort({namelist.name});
namelist_order = namelist(index);

%%%%%%%%%%%%%%%%%%%%%%%%
start_num = 1;
testnumber = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3. surface reconsterution by fitting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3.1 for each surface %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters in the algorithms
maxDistance = 0.05;
imageSize = [227 227 3]; % the input image should be pre-processed into same size

% for i = start_num: start_num+ testnumber% len % = 103
for i = 1: len % len % = 103
    % load the points in the test file
    tic % count time

    P_C_target = importdata(namelist_order(i).name); % L is the 3D point cloud
    size(P_C_target)
    image_process = process_point_cloud(P_C_target,i); % return the obtianed figure

    imds = augmentedImageDatastore(imageSize, image_process);
    [YPred,scores] = classify(netTransfer,imds); % predict
    str = cellstr(YPred);
    strtemp = str{1}; % 1 plane 2 cylidner 3 sphere 4 cone 5 sphere

    % visulize the case

    % for fitting the points
    if size(P_C_target,1) > 8000 % when the data is big and it contains noise, there could be problem
        y = randsample(size(P_C_target,1),5000);
        P_C_target = P_C_target(y,:);
    end 
    ptCloud = pointCloud(P_C_target);
    center_point = mean(P_C_target);

    % figure;
    % scatter3(P_C_target(:,1),P_C_target(:,2),P_C_target(:,3),'.');
    % axis equal;
    % set(get(gca, 'Title'), 'String', strtemp);
    % hold on

    switch strtemp % the result is quite close to the visualization 
        case '1' % plane fitting
            model = pcfitplane(ptCloud,maxDistance);
            
            % % visulize the plane 
            plot(model);
            figure
            scatter3(P_C_target(:,1),P_C_target(:,2),P_C_target(:,3),'.');
            axis equal;
            hold on;
            surface_infor = [1 model.Normal center_point]'
            x0 = center_point; 
            w = null(model.Normal); % Find two orthonormal vectors which are orthogonal to a (row vector)
            [P,Q] = meshgrid(-8:8); % Provide a gridwork (you choose the size)
            X = x0(1)+w(1,1)*P+w(1,2)*Q; % Compute the corresponding cartesian coordinates
            Y = x0(2)+w(2,1)*P+w(2,2)*Q; %   using the two vectors in w
            Z = x0(3)+w(3,1)*P+w(3,2)*Q;
            surf(X,Y,Z)
        case '2' % cylidner fitting
            model = pcfitcylinder(ptCloud,maxDistance);
            % % visulize the cylidner 
            % plot(model);
            surface_infor = [2 model.Radius  model.Orientation/norm(model.Orientation) model.Center]'
        case '3' % sphere fitting 
            [Center,Radius] = sphereFit(P_C_target);
            % visulize the sphere
            a = Center(1); b = Center(2); c = Center(3); r = Radius;    
            %%
            % [x,y,z] = sphere;
            % x = x*r;
            % y = y*r;
            % z = z*r;
            % surf(x+a,y+b,z+c)
            surface_infor = [3 Radius Center]'
        case '4' % cone fitting 
            [surface_infor, flag] = cone_fitting (P_C_target, ptCloud)
            if flag == 2
                %cylinder fitting
                model = pcfitcylinder(ptCloud,maxDistance);
                if model.Radius < 20 % not a big cylidner
                    % plot(model)
                    surface_infor = [2 model.Radius  model.Orientation/norm(model.Orientation) model.Center]'
                else
                    flag = 1;
                end
            end
            if flag == 1
                %plane fitting
                model = pcfitplane(ptCloud,maxDistance);
                % % visulize the cylidner 
                % plot(model)
                surface_infor = [1 model.Normal center_point]'
            end

        case '5' % torus fitting
            [x0n, an, rn, sn, flag] = torus_fiiting(P_C_target);
            if flag == 5
                % % visulize the torus 
                % see_torus(rn, sn, x0n', an');
                surface_infor = [5 rn sn  an' x0n']'
            elseif flag == 3 % mislabeled sphere case
                [Center,Radius] = sphereFit(P_C_target);
                % visulize the sphere
                a = Center(1); b = Center(2); c = Center(3); r = Radius;    
                % %%
                % [x,y,z] = sphere;
                % x = x*r;
                % y = y*r;
                % z = z*r;
                % surf(x+a,y+b,z+c)
                surface_infor = [3 Radius Center]'
            else % mislabeled cylinder case
                model = pcfitcylinder(ptCloud,maxDistance);
                if model.Radius < 20 % not a big cylidner
                    % plot(model)
                    surface_infor = [2 model.Radius  model.Orientation/norm(model.Orientation) model.Center]'
                else
                    %plane fitting
                    model = pcfitplane(ptCloud,maxDistance);
                    % % visulize the cylidner 
                    % plot(model)
                    surface_infor = [1 model.Normal center_point]'
                end
            end
    end
    % set(get(gca, 'Title'), 'String', surface_infor(1));
    t(i)=toc;

    pause

   
    % file_name =  ['D:\matlab workplace\partition\Measurement test\prediction_result\pointCloud', num2str(i), '_prediction.txt'];
    % fid=fopen(file_name,'wt');
    % fprintf(fid,'%s\n',num2str(surface_infor(1)));
    % for i = 2:length(surface_infor)
    %     fprintf(fid,'%.16f\n',surface_infor(i));
    % end
    % fclose(fid);

end

t

function im = process_point_cloud(Point_cloud,i)

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
    h_fig = figure('Visible', 'off');        
    scatter3(Point_cloud(:,1), Point_cloud(:,2), Point_cloud(:,3), 50, indexcolor,'filled');
    axis equal
    axis off
    view(0,0)
    % save as png file
    frame = getframe(gcf);
    im = frame2im(frame);
    close(h_fig);

end

function [Center,Radius] = sphereFit(X)
% this fits a sphere to a collection of data using a closed form for the
% solution (opposed to using an array the size of the data set). 
% Minimizes Sum((x-xc)^2+(y-yc)^2+(z-zc)^2-r^2)^2
% x,y,z are the data, xc,yc,zc are the sphere's center, and r is the radius

% Assumes that points are not in a singular configuration, real numbers, ...
% if you have coplanar data, use a circle fit with svd for determining the
% plane, recommended Circle Fit (Pratt method), by Nikolai Chernov
% http://www.mathworks.com/matlabcentral/fileexchange/22643

% Input:
% X: n x 3 matrix of cartesian data
% Outputs:
% Center: Center of sphere 
% Radius: Radius of sphere
% Author:
% Alan Jennings, University of Dayton

A=[mean(X(:,1).*(X(:,1)-mean(X(:,1)))), ...
    2*mean(X(:,1).*(X(:,2)-mean(X(:,2)))), ...
    2*mean(X(:,1).*(X(:,3)-mean(X(:,3)))); ...
    0, ...
    mean(X(:,2).*(X(:,2)-mean(X(:,2)))), ...
    2*mean(X(:,2).*(X(:,3)-mean(X(:,3)))); ...
    0, ...
    0, ...
    mean(X(:,3).*(X(:,3)-mean(X(:,3))))];
A=A+A.';
B=[mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,1)-mean(X(:,1))));...
    mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,2)-mean(X(:,2))));...
    mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,3)-mean(X(:,3))))];
Center=(A\B).';
Radius=sqrt(mean(sum([X(:,1)-Center(1),X(:,2)-Center(2),X(:,3)-Center(3)].^2,2)));

end


function see_torus(rn, sn, x0n, a0)
    
    aminor = sn; % Torus minor radius
    Rmajor = rn; % Torus major radius
    
    theta  = linspace(-pi, pi, 64)   ; % Poloidal angle
    phi    = linspace(0., 2.*pi, 64) ; % Toroidal angle

    [t, p] = meshgrid(phi, theta);

    % move to the center point
    x = (Rmajor + aminor.*cos(p)) .* cos(t) ;
    y = (Rmajor + aminor.*cos(p)) .* sin(t);
    z = aminor.*sin(p);

    points = [x(:) y(:) z(:)];
    
    v1= [0 0 1]; % original orientation of the cylinder
    v2= a0; % z direction

    % find the rotation metrix of the point cloud
    nv1 = v1/norm(v1);
    nv2 = v2/norm(v2);

    if norm(nv1+nv2)==0
        q = [0 0 0 0];
    else
        u = cross(nv1,nv2);         
        u = u/norm(u);
        theta = acos(sum(nv1.*nv2))/2;
        q = [cos(theta) sin(theta)*u];
    end

    %rotation metrix
    R=[2*q(1).^2-1+2*q(2)^2  2*(q(2)*q(3)+q(1)*q(4)) 2*(q(2)*q(4)-q(1)*q(3));
        2*(q(2)*q(3)-q(1)*q(4)) 2*q(1)^2-1+2*q(3)^2 2*(q(3)*q(4)+q(1)*q(2));
        2*(q(2)*q(4)+q(1)*q(3)) 2*(q(3)*q(4)-q(1)*q(2)) 2*q(1)^2-1+2*q(4)^2];

    % rotate the cloud point so that the direction becomes [0 0 1]
    points = points*R + x0n;
    scatter3(points(:,1), points(:,2), points(:,3), 5, 'r');
    axis equal

end

function [x0n, an, rn, sn, flag] = torus_fiiting(P_C_target)

    flag = 5; 

    [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [1 0 1]', 2, 1, 1, 2);
    recordvalue(1,:) = {x0n, an, rn, sn};
    con_1 = conv;
    sigmah_record(1) = sigmah;

    [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [0 0 1]', 2, 1, 1, 2);
    recordvalue(2,:) = {x0n, an, rn, sn};
    con_2 = conv;
    sigmah_record(2) = sigmah;

    [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [0 1 0]', 2, 1, 1, 2);
    recordvalue(3,:) = {x0n, an, rn, sn};
    con_3 = conv;
    sigmah_record(3) = sigmah;

    [~,index] = min(sigmah_record);

    % is three direction are not all conveged, there might not be a way to find the right position
    % maybe more steps
    if min(sigmah_record) > 0.1 %not converged well
        [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [1 0 1]', 2, 1, 0.01, 0.02);
        recordvalue(1,:) = {x0n, an, rn, sn};
        con_1 = conv;
        sigmah_record(1) = sigmah;
    
        [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [0 0 1]', 2, 1, 0.01, 0.02);
        recordvalue(2,:) = {x0n, an, rn, sn};
        con_2 = conv;
        sigmah_record(2) = sigmah;
    
        [x0n, an, rn, sn, d, sigmah, conv, Vx0n, Van, urn, usn, GNlog, a, R0, R] = lstorus(P_C_target, [0 0 0]', [0 1 0]', 2, 1, 0.01, 0.02);
        recordvalue(3,:) = {x0n, an, rn, sn};
        con_3 = conv;
        sigmah_record(3) = sigmah;
    
        [~,index] = min(sigmah_record);
    end 

    x0n = recordvalue{index, 1};
    an = recordvalue{index, 2};
    rn = recordvalue{index, 3};
    sn = recordvalue{index, 4};
    convall =  con_1+ con_2+con_3;

    if convall == 0
            % % return a unit torus
            % rn = 2;
            % sn = 1;
            % x0n = [0 0 0]';
            % an = [0 0 1]';
            flag = 2;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% when some index is reached, we need to change the surface type %%%%%%%%%%%%%%%%%%%%%%%
    if abs(rn) < 0.125 % mislabeled sphere 
        flag = 3;
    end

    if (rn < 1.1*sn) & convall & (abs(rn) > 0.5) % move the second case to cylinder
        flag = 2;
    end
end

function [surface_infor, flag] = cone_fitting (P_C_target, ptCloud)

    flag = 4; % flag of cone
    surface_infor = [];
    fobj=rightcircularconeFit(P_C_target') ;%perform the fit
    % if it is not a cone, the fitting will be bad, then we change the fitting
    %Visualize the fit

    if ~isreal(fobj.cone_angle)
        flag = 1; % surface_type = 'plane'
        return;
    end

    % plot(fobj) ;
    % xlabel X; ylabel Y; zlabel Z;
    % xlim([-10 10]); ylim([-10 10]); zlim([-10 10]);
    % axis vis3d

    if (fobj.cone_angle < 2) | (fobj.cone_angle > 90)  % small angle maybe bad fitting, maybe cylinder

        % denoise the point cloud 3 times 
        for k = 1:3
            ptCloud = pcdenoise(ptCloud);
        end
        P_C_target_view1 = ptCloud.Location;
        % size(P_C_target_view1);

        % figure; % denoise
        % scatter3(P_C_target_view1(:,1),P_C_target_view1(:,2),P_C_target_view1(:,3),'.');
        % axis equal;

        fobj=rightcircularconeFit(P_C_target_view1'); %perform the fit
        if ~isreal(fobj.cone_angle)
            flag = 1;
            return;
        end

        if (fobj.cone_angle>30) & (fobj.cone_angle < 70)
            % surface_type = 'cone no need calculation'
            % hold on
            % plot(fobj) ;
            % xlabel X; ylabel Y; zlabel Z;
            % xlim([-10 10]); ylim([-10 10]); zlim([-10 10]);
            % axis vis3d

            cone_axis = rotz(fobj.yaw)* roty(fobj.pitch)*[1 0 0]';
            surface_infor = [4 fobj.cone_angle*pi/180 cone_axis(1) cone_axis(2) cone_axis(3) fobj.vertex]'
            return;
        end

        
        [x0n, an, phin, rn, d, sigmah, conv, Vx0n, Van, uphin, urn, GNlog, a, R0, R] = lscone(P_C_target_view1, mean(P_C_target_view1)', [0 0 1]', pi/4, 2, 0.01, 0.02);
        angle = mod(phin*180/pi,90);
        recordvalue(1,:) = {x0n, an, phin, angle, rn};
        sigmah_record(1) = sigmah;
        con_1 = conv;

        [x0n, an, phin, rn, d, sigmah, conv, Vx0n, Van, uphin, urn, GNlog, a, R0, R] = lscone(P_C_target_view1, mean(P_C_target_view1)', [0 1 0]', pi/4, 2, 0.01, 0.02);
        angle = mod(phin*180/pi,90);
        recordvalue(2,:) = {x0n, an, phin, angle, rn};
        sigmah_record(2) = sigmah;
        con_2 = conv;


        [x0n, an, phin, rn, d, sigmah, conv, Vx0n, Van, uphin, urn, GNlog, a, R0, R] = lscone(P_C_target_view1, mean(P_C_target_view1)', [1 0 0]', pi/4, 2, 0.01, 0.02);
        angle = mod(phin*180/pi,90);
        recordvalue(3,:) = {x0n, an, phin, angle, rn};
        sigmah_record(3) = sigmah;
        con_3 = conv;


        [B, II] = mink(sigmah_record,2);
        temp_v = B(2) - B(1);
        if (temp_v ~= 0) &  temp_v < 0.01
            % at this special situation (the two local minimum is not far)
            % we comepare the angle
            if recordvalue{II(1), 4} < recordvalue{II(2), 4}
                index = II(1);
            else
                index = II(2);
            end
        else
            index = II(1);
        end
        x0n = recordvalue{index, 1};
        an = recordvalue{index, 2};
        phin = recordvalue{index, 3};
        angle = recordvalue{index, 4};
        rn = recordvalue{index, 5};
        convall =  con_1+ con_2+con_3;

        d  = zeros(2,3);
        d(1,:) = x0n'+ (rn/tan(phin))*an';
        d(2,:) = x0n'-2*an';


         % I give up. let's say it is cylinder
         if  (angle < 25) | (angle > 75)
            if angle < 20
                flag = 2 ; % cylnider
                % if cylinder radii is too big ,it is a plane
                return;
            end
            if angle > 75
                flag = 2; % cylidner
                % if cylidner radii is too big ,it is a plane
                return;
            end

        end

        if convall== 0 % no convge
            flag = 2; % cylidner
            % if cylidner radii is too big ,it is a plane
            if (angle < 60) & (angle > 30)
                flag = 4; %cone
                surface_infor = [4 phin an(1) an(2) an(3) d(1,:)]';
            end
            return;
        end

        % [X3,Y3,Z3,X,Y,Z] = cone(angle,d,5);

        % figure;
        % scatter3(P_C_target(:,1),P_C_target(:,2),P_C_target(:,3),'.');
        % axis equal;
        % hold on
        % surf(X3+d(1,1),Y3+d(1,2),Z3+d(1,3))

    end

    cone_axis = rotz(fobj.yaw)* roty(fobj.pitch)*[1 0 0]';
    surface_infor = [4 fobj.cone_angle*pi/180 cone_axis(1) cone_axis(2) cone_axis(3) fobj.vertex]';

end


function [X3,Y3,Z3,X,Y,Z] = cone(theta,d,h)
    %    theta = 45;
        
        r = h*tan(pi*theta/180);
    
        m = h/r;
        [R,A] = meshgrid(linspace(0,r,11),linspace(0,2*pi,41));
        % Generate cone about Z axis with given aperture angle and height
        X = R .* cos(A);
        Y = R .* sin(A);
        Z = m*R;
        % Cone around the z-axis, point at the origin
        % find coefficients of the axis vector xi + yj + zk
        x = d(2,1)-d(1,1);
        y = d(2,2)-d(1,2);
        z = d(2,3)-d(1,3);
        
        % find angle made by axis vector with X axis
        phix = atan2(y,x);
        % find angle made by axis vector with Z axis
        phiz = atan2(sqrt(x^2 + y^2),(z));
        
        % Rotate once about Z axis 
        X1 = X*cos(phiz)+Z*sin(phiz);
        Y1 = Y;
        Z1 = -X*sin(phiz)+Z*cos(phiz);
    
        % Rotate about X axis
        X3 = X1*cos(phix)-Y1*sin(phix);
        Y3 = X1*sin(phix)+Y1*cos(phix);
        Z3 = Z1;

        
end