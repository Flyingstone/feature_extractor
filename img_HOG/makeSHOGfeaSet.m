function feaSet = makeSHOGfeaSet(feat,imWidth,imHeight,HOGCellSize)
if nargin < 4
    HOGCellSize = 8;
end

[gridYSize, gridXSize, Dim_HOG] = size(feat);
remX = mod(imWidth,HOGCellSize);
remY = mod(imHeight,HOGCellSize);

remXHOG = (imWidth - remX) / HOGCellSize - gridXSize;
remXHOG = remXHOG / 2;
remYHOG = (imHeight - remY) / HOGCellSize - gridYSize;
remYHOG = remYHOG / 2;

remXAll = remX / 2 + remXHOG*HOGCellSize;
remYAll = remY / 2 + remYHOG*HOGCellSize;

feaSet.feaArr = zeros(Dim_HOG*4,(gridXSize-1)*(gridYSize-1),'double');
feaSet.x = zeros((gridXSize-1)*(gridYSize-1),1,'double');
feaSet.y = zeros((gridXSize-1)*(gridYSize-1),1,'double');

counter = 1;
for x=1:gridXSize-1
    for y=1:gridYSize-1
        feaSet.x(counter) = remXAll + x*HOGCellSize;
        feaSet.y(counter) = remYAll + y*HOGCellSize;
        temp = [reshape(feat(y,x,:),Dim_HOG,1); reshape(feat(y,x+1,:),Dim_HOG,1); reshape(feat(y+1,x,:),Dim_HOG,1); reshape(feat(y+1,x+1,:),Dim_HOG,1)];
        feaSet.feaArr(:,counter) = temp(:);
        counter = counter + 1;
    end
end
feaSet.width = imWidth;
feaSet.height = imHeight;



