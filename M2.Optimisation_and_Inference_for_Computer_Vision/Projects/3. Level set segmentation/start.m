%close all;
clearvars;
clc

% I=double(imread('zigzag_mask.png'));
% I=double(imread('Image_to_Restore.png'));
% I=mean(I,3); %To 2D matrix
% I=double(imread('circles.png'));
% I=double(imread('noisedCircles.tif'));
% I=double(imread('phantom17.bmp'));
% I=double(imread('phantom18.bmp'));
% I=double(imread('our_images/scissors_noise.png'));
% I=double(imread('our_images/scissors.png'));
I=double(imread('our_images/rabbit.png'));
% I=double(imread('our_images/oranges_noise.png'));
% I=double(imread('our_images/oranges.png'));
% I=double(imread('our_images/heart.png'));
% I=double(imread("our_images/circles.png"));
I=mean(I,3);
% I = I(:,:,1);
I=I-min(I(:));
I=I/max(I(:));

[ni, nj]=size(I);



%Lenght and area parameters
%circles.png mu=1, mu=2, mu=10
%noisedCircles.tif mu=0.1
%phantom17 mu=1, mu=2, mu=10
%phantom18 mu=0.2 mu=0.5
%hola carola mu=1
mu=0.1;
nu= 0.0;

%%Parameters
% lambda1=1;
% lambda2=1;
lambda1=1; %Hola carola problem
lambda2=1; %Hola carola problem
 
epHeaviside=1;
eta=1;
tol=0.01;
dt=(10^-1)/mu;
iterMax=100000;
reIni=0;
[X, Y]=meshgrid(1:nj, 1:ni);

%%Initial phi
% 
% phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);
% phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/4)).^2)+50);
phi_0 = checkerboard(ni, nj, 20);
% phi_0=I; %For the Hola carola problem

%Normalization of the initial phi to [-1 1]
phi_0=phi_0-min(phi_0(:));
phi_0=2*phi_0/max(phi_0(:));
phi_0=phi_0-1;



%%Explicit Gradient Descent
seg=sol_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni );

