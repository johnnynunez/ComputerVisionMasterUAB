clearvars;
DST_NAME = 'cars.png';
SRC_NAME = 'karim.png';

dst = double(imread(DST_NAME));
src = double(imread(SRC_NAME)); % flipped girl, because of the eyes
[ni,nj, nChannels]=size(dst);

param.hi=1;
param.hj=1;


%masks to exchange: Eyes
mask_dst=logical(imread(strcat('mask_dst_', DST_NAME)));
mask_src=logical(imread(strcat('mask_src_', SRC_NAME)));

for nC = 1: nChannels
    
    %TO DO: COMPLETE the ??
    
    % Second order central diference (second parcial derivatives)
    drivingGrad_i = sol_DiBwd( src(:,:,nC), param.hi ) - sol_DiFwd( src(:,:,nC), param.hi );
    drivingGrad_j = sol_DjBwd( src(:,:,nC), param.hj ) - sol_DjFwd( src(:,:,nC), param.hj );
    
    % Laplacian
    driving_on_src = drivingGrad_i + drivingGrad_j;

    driving_on_dst = zeros(size(dst(:,:,1)));   
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));
    
    param.driving = driving_on_dst;

    dst1(:,:,nC) = sol_Poisson_Equation_Axb(dst(:,:,nC), mask_dst,  param);
end

%Mouth
%masks to exchange: Mouth
% mask_src=logical(imread('mask_src_mouth.png'));
% mask_dst=logical(imread('mask_dst_mouth.png'));
% 
% for nC = 1: nChannels
%     
%     % Second order central diference (second parcial derivatives)
%     drivingGrad_i =  sol_DiBwd( src(:,:,nC), param.hi ) - sol_DiFwd( src(:,:,nC), param.hi );
%     drivingGrad_j =  sol_DjBwd( src(:,:,nC), param.hj ) - sol_DjFwd( src(:,:,nC), param.hj );
%     
%     % Laplacian
%     driving_on_src = drivingGrad_i + drivingGrad_j;
% 
%     driving_on_dst = zeros(size(src(:,:,1)));  
%     driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));
%     
%     param.driving = driving_on_dst;
% 
%     dst1(:,:,nC) = sol_Poisson_Equation_Axb(dst1(:,:,nC), mask_dst,  param);
%  end

imwrite(dst1/256, strcat('img_poisson_',DST_NAME))
imshow(dst1/256)

