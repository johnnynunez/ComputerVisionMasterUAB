function [ result ] = get_max_grad( grad_dst, grad_src, mask_dst, mask_src)    
% Compute the Forward finite differences with respect to the
% i coordinate only for the 1:end-1 rows. The last row is not replace
    
    grad_dst_masked = grad_dst(mask_dst(:));
    grad_src_masked = grad_src(mask_src(:));
    grad_mask = abs(grad_dst_masked) <= abs(grad_src_masked);
    grad_dst_masked(grad_mask) = grad_src_masked(grad_mask);
    grad_dst(mask_dst(:)) = grad_dst_masked;
    result = grad_dst;
end