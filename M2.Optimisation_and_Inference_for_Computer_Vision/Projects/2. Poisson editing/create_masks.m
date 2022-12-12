
clearvars;
dst_file = 'dembele';
src_file = 'glasses';


dst = double(imread(strcat(dst_file, '.png')));
src = double(imread(strcat(src_file, '.png'))); % smaller pls

mask_src = zeros(size(src(:,:,1)));
mask_dst = zeros(size(dst(:,:,1)));

w = 107; % size mask altura
h = 512; % anchura
init_w = 190; % position in the image altura
init_h = 140; % anchura
for i = init_w:w+init_w
    for j = init_h:h+init_h
        mask_src(i, j) = 1;
    end
end


destination_t = 1155; % position in the image altura
destination_l = 1600; % anchura
for i = destination_t:w+destination_t
    for j = destination_l:h+destination_l
        mask_dst(i, j) = 1;
    end
end

figure(1)
imshow(mask_src)
figure(2)
imshow(mask_dst)


imwrite(mask_src, strcat('mask_src_', src_file, '.png'))
imwrite(mask_dst, strcat('mask_dst_', dst_file, '.png'))
