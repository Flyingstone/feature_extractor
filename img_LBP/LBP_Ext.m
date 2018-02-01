function feaArr = makeLBPSP(LBPImage,bins)

[h,w,c] = size(LBPImage);

wsub = w / 4;
hsub = h / 4;

% feaArr = zeros(bins*21,1,'int8');
base3 = zeros(bins,4,4,'single');

for gridX = 1:4
    for gridY = 1:4
        tempI = LBPImage(((gridY-1)*hsub+1):gridY*hsub,((gridX-1)*wsub+1):gridX*wsub);
        result=hist(tempI(:),0:(bins-1));
        base3(:,gridY,gridX) = result(:) / sum(result);
    end
end
base2 = zeros(bins,2,2,'single');
for gridX = 1:2
    for gridY = 1:2
        base2(:,gridY,gridX) = base3(:,gridY*2-1,gridX*2-1)+base3(:,gridY*2,gridX*2-1)+base3(:,gridY*2-1,gridX*2)+base3(:,gridY*2,gridX*2);
        base2(:,gridY,gridX) =  base2(:,gridY,gridX) / sum( base2(:,gridY,gridX));
    end
end
base = base2(:,1,1)+base2(:,1,2)+base2(:,2,1)+base2(:,2,2);
base(:) = base(:)/sum(base(:));

feaArr = [base3(:); base2(:); base(:)];