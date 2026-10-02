function mpi = sai2mpi(sai,HWUV)
% Sub-Aperture Images to Macro-Pixel Image function
% Created in 2021/03/08 by kimchange
% HWUV: U*V Sub-Aperture Images of size Height*Width

if nargin==1 % num of args in is 1 (sai)
    assert(length(size(sai)) == 4 | (length(size(sai)) == 3 & sqrt(size(sai, 3)) == floor( sqrt(size(sai, 3))) ), 'lack HWUV info of sai');
    HWUV = size(sai);
    if(length(HWUV) == 3)
        HWUV(3) = sqrt(size(sai, 3)); HWUV(4) = HWUV(3);
        H = HWUV(1); W = HWUV(2); U = HWUV(3); V = HWUV(4);
        sai = reshape(sai, [H, W, V, U]);
        sai = permute(sai, [1, 2, 4, 3]);
    end
    H = HWUV(1); W = HWUV(2); U = HWUV(3); V = HWUV(4);
else
    assert(length(size(sai)) == 2 & length(HWUV) == 4, 'wrong format');
    H = HWUV(1); W = HWUV(2); U = HWUV(3); V = HWUV(4);
    sai = reshape(sai, [H, U, W*V]);
    sai = permute(sai, [1,3,2]); % now [H, W*V, U];
    sai = reshape(sai, [H, W, V, U]);
    sai = permute(sai, [1, 2, 4, 3]);
end

mpi = sai;
mpi = permute(mpi, [3, 1, 4, 2]);
mpi = reshape(mpi, [U*H, V*W]);

end
