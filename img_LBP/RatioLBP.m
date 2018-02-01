function Rel=RatioLBP(ImgData)

hwinsize=1;

% get the image height and width
[height, width]=size(ImgData);

% get the point
spoints=[-1 -1; -1 0; -1 1; 0 -1; -0 1; 1 -1; 1 0; 1 1];

num_p=size(spoints,1);

% Center pixels Coordinates of origin (0,0) in the Image
origy=2;
origx=2;
% Calculate dx and dy;
dx = width-3;
dy = height-3;

% the center image data
C = ImgData(origy:origy+dy, origx:origx+dx);
Pdata=zeros(dy+1, dx+1, num_p);

for i=1:num_p
    x = spoints(i,1)+origx;
    y = spoints(i,2)+origy;
    rx=round(x);
    ry=round(y);
    tmpData=ImgData(ry:ry+dy, rx:rx+dx);
    tmpData=tmpData./C;
    Pdata(:,:,i) = tmpData>1;
end

% Get the LBP data of each pixels
Rel=zeros(dy+1, dx+1);
Rel = Pdata(:,:,1)+Pdata(:,:,2).*2+Pdata(:,:,3).*4+Pdata(:,:,4).*8+...
    Pdata(:,:,5).*16+Pdata(:,:,6).*32+Pdata(:,:,7).*64+Pdata(:,:,8).*128;

% process the edge of the image
Rel = [Rel(1,:); Rel; Rel(dy+1, :)];
Rel = [Rel(:,1) Rel Rel(:,dx+1)];

end