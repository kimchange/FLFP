function Backprojection = backwardProject(H, projection)

    if length(size(H) ) == 4
        Nnum = sqrt(size(H,3));
        H = reshape(H, [size(H,1), size(H,2), Nnum, Nnum, size(H,4)]);

    end

    Nnum = size(H,3);
    x3length = size(H,5);
    Backprojection = zeros(size(projection, 1), size(projection, 1), x3length );
    
    
    for aa=1:Nnum,
        for bb=1:Nnum,
            % disp([aa, bb])
            for cc=1:x3length,
                          
                Ht = imrotate( H(:,:,aa,bb,cc), 180);            
                % tempSlice = conv2(projection, Ht, 'same');     
                tempSlice = conv2fft_gpu(projection, Ht);          
                Backprojection((aa:Nnum:end) , (bb:Nnum:end),cc) = Backprojection((aa:Nnum:end) , (bb:Nnum:end),cc) + tempSlice( (aa:Nnum:end) , (bb:Nnum:end) );
                
            end
        end
    end

end