function a = pca_plane(pc)

%% Rotate the point cloud along x and y direction
a =1;

direction = 'x';
pcx = rotatePointCloudAlongZ(pc, direction);

direction = 'y';
pcy = rotatePointCloudAlongZ(pc, direction);

%% 

figure
plot3(pc(:, 1), pc(:, 2), pc(:, 3), 'b.')
hold on
centerpoint = mean(pc)
V = pca(pc);

% t= 1: 0.05 :10;
% x =  V(1,1)*t +centerpoint(1);
% y =  V(1,2)*t + centerpoint(2);
% z =  V(1,3)*t + centerpoint(3);

% scatter3(x, y, z);

% t= 1: 0.05 :10;
% x =  V(2,1)*t +centerpoint(1);
% y =  V(2,2)*t + centerpoint(2);
% z =  V(2,3)*t + centerpoint(3);

% scatter3(x, y, z);

% t= 1: 0.05 :10;
% x =  V(3,1)*t +centerpoint(1);
% y =  V(3,2)*t + centerpoint(2);
% z =  V(3,3)*t + centerpoint(3);

% scatter3(x, y, z);



% t= 1: 0.05 :10;
% x =  V(1,1)*t +centerpoint(1);
% y =  V(2,1)*t + centerpoint(2);
% z =  V(3,1)*t + centerpoint(3);

% scatter3(x, y, z);

% t= 1: 0.05 :10;
% x =  V(1,2)*t +centerpoint(1);
% y =  V(2,2)*t + centerpoint(2);
% z =  V(3,2)*t + centerpoint(3);

% scatter3(x, y, z);

% t= 1: 0.05 :10;
% x =  V(1,3)*t +centerpoint(1);
% y =  V(2,3)*t + centerpoint(2);
% z =  V(3,3)*t + centerpoint(3);

% scatter3(x, y, z);

quiver3(centerpoint(1),centerpoint(2),centerpoint(3),V(1, 1),V(2, 1),V(3, 1), 'g', 'LineWidth', 4);
quiver3(centerpoint(1),centerpoint(2),centerpoint(3),V(1, 2),V(2, 2),V(3, 2), 'k', 'LineWidth', 4);
quiver3(centerpoint(1),centerpoint(2),centerpoint(3),V(1, 3),V(2, 3),V(3, 3), 'r', 'LineWidth', 4);
axis equal
grid on; grid minor; box on;
xlabel('x'); ylabel('y'); zlabel('z')
view(3)
title('Arrows are PCA eigenvectors')