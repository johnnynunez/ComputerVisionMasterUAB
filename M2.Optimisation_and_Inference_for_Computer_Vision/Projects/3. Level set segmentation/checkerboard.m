function result = checkerboard( img_size_i, img_size_j, square_size)
%  Generates a checkerboard level set function.
%    According to Pascal Getreuer, such a level set function has fast
%    convergence.

xv = 1:img_size_j;
yv = (1:img_size_i)';
sf = pi / square_size;
xv = xv .* sf;
yv = yv .* sf;
result = sin(xv) .* sin(yv);