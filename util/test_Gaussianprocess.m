clear all;
clc;

% % Example 1:
%   % build the correlation struct
%   corr.name = 'gauss';
%   corr.c0 = 1;
%   corr.sigma = 1;

%   mesh = linspace(-1,1,101)';              % generate a mesh
%   data.x = [-1; 1]; data.fx = [0; -1];    % specify boundaries

%   [F,KL] = randomfield(corr, mesh, ...
%               'nsamples', 10, ...
%               'data', data, ...
%               'filter', 0.95);

%   % to generate 100 more samples using the KL
%   trunc = length(KL.sv);                  % get the truncation level
%   W = randn(trunc,100); 
%   F2 = repmat(KL.mean,1,100) + KL.bases*diag(KL.sv)*W;

% % Example 2 (2-D):
%   % build the correlation struct
%   corr.name = 'exp';
%   corr.c0 = [0.2 1]; % anisotropic correlation

%   x = linspace(-1,1,28);
%   [X,Y] = meshgrid(x,x); mesh = [X(:) Y(:)]; % 2-D mesh

%   % set a spatially varying variance (must be positive!)
%   corr.sigma = cos(pi*mesh(:,1)).*sin(2*pi*mesh(:,2))+1.5;

%   [F,KL] = randomfield(corr,mesh,...
%               'trunc', 10);

%   % plot the realization
%   surf(X,Y,reshape(F,28,28)); view(2); colorbar;

    % build the correlation struct
    corr.name = 'gauss';
    corr.c0 = 1; % isotropic correlation
  
    x = linspace(-1,1,28);
    [X,Y] = meshgrid(x,x); mesh = [X(:) Y(:)]; % 2-D mesh
  
    % set a spatially varying variance (must be positive!)
    corr.sigma = 0.02;
  
    [F,KL] = randomfield(corr,mesh,...
                'trunc', 10);
  
    % plot the realization
    surf(X,Y,reshape(rand()*0.1*F,28,28)); view(3); colorbar;