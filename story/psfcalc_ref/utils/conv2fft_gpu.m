function y = conv2fft_gpu(x, kern)
    % size(y) == size(x), size(kern,1) == size(kern, 2)
    [h,w] = size(x);
    [hk,wk] = size(kern);
    assert(hk==wk & mod(hk,2) == 1,'size(kern,1) not an odd num or not equal size(kern, 2) ');
    r = h + hk -1;
    x_ = gpuArray.zeros(r,r,'single');
    kern_ = gpuArray.zeros(r,r,'single');
    y_ = gpuArray.zeros(r,r,'single');
    x_(1:h, 1:w) = x;
    kern_(1:hk, 1:wk) = kern;
    y_ = real( ifft2(fft2(x_) .* fft2(kern_)) );
    y = y_((hk+1)/2 : (hk+1)/2 + h -1, (hk+1)/2 : (hk+1)/2 + w -1 );
    end