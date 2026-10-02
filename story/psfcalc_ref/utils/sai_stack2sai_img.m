function sai_img = sai_stack2sai_img(sai_stack,UV)
% sai_stack2sai_img transforms Sub-Aperture Images Stack to Sub-Aperture Images in a big slice function
% Created in 2021/03/08 by kimchange
% UV: U*V Sub-Aperture Images 
% [H, W, U, V] TO [H*U. W*V] 
% OR [H, W, U*U] TO [H*U. W*V] when U==V

if nargin==1 % num of args in is 1 (sai)
    assert(length(size(sai_stack)) == 4 | (length(size(sai_stack)) == 3 & sqrt(size(sai_stack, 3)) == floor( sqrt(size(sai_stack, 3))) ), 'lack UV info of sai');
    HWUV = size(sai_stack);
    if(length(HWUV) == 3)  % U==V
        HWUV(3) = sqrt(size(sai_stack, 3)); HWUV(4) = HWUV(3);
        H = HWUV(1); W = HWUV(2); U = HWUV(3); V = HWUV(4);
        sai_stack = reshape(sai_stack, [H, W, V, U]);
        sai_stack = permute(sai_stack, [1, 2, 4, 3]);
    end
    H = HWUV(1); W = HWUV(2); U = HWUV(3); V = HWUV(4);
else
    U = UV(1); V = UV(2);
    assert( length(size(sai)) == 4 | (length(size(sai)) == 3 & size(sai,3) == U*V), 'wrong format');
    H = size(sai,1); W = size(sai,2);
    if(length(HWUV) == 3)
        sai_stack = reshape(sai_stack, [H, W, V, U]);
        sai_stack = permute(sai_stack, [1, 2, 4, 3]);
    end
end

sai_img = sai_stack;
sai_img = permute(sai_img, [1, 3, 2, 4]);
sai_img = reshape(sai_img, [H*U, W*V]);

end