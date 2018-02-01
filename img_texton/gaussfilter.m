function result = gaussfilter(image)
% Convolves image with 36 filters at 6 orientations,
% 3 scales and 2 phases that have an aspect ration of 3:1
% e.g. gaussfilter(I, 'penguin'), I = imread('penguin','jpg')

[m,n] = size(image); % Make filter sizes dependent on image size

% set scale
sigma = round(0.01*(sqrt(m.^2 + n.^2)));
%sigma = 0.01 * length_diag

i = 0;

% Create platform for filter
%sigma used to be equal to 1, independent of images size
X = [-5*sigma:5*sigma];
%X = [-25*sigma:25*sigma]; % for drawing filters
Y = X;
[x, y] = meshgrid(X, Y);


% Rotate coordinates on which filter is built
for rot = 0:5
    theta = pi/6.*[rot];	% orientations 
    X = cos(theta)*x + sin(theta)*y;
    Y = -sin(theta)*x + cos(theta)*y;
    
    % Use three different scales for filters
    % sigy will be equal to sigma above, then increase in half octave steps
    for scale = log2(sigma):.5:log2(sigma)+1 % three 1/2 octave steps
        sigy = 2^(scale);
        sigx = 3*sigy;
        
        gauss = 1/(sqrt(2*pi)*sigx) * exp((-X.^2)/(2*sigx^2));
        gauss1 = -Y./(sqrt(2*pi)*sigy^3) .* exp((-Y.^2)/(2*sigy^2));
        gauss2 = ((Y.^2/(sqrt(2*pi)*sigy^5)) - (1/(sqrt(2*pi)*sigy^3))) .* exp((-Y.^2)/(2*sigy^2));
        filter1 = (gauss.*gauss1);
        filter2 = (gauss.*gauss2);
        
        %L1 normalize
        filter1 = (filter1 - mean(filter1(:)));
        filter2 = (filter2 - mean(filter2(:)));
        filter1 = filter1/sum(abs(filter1(:)));
        filter2 = filter2/sum(abs(filter2(:)));
        
        
        if 0 % viewfilters
            i = i + 1;
            figure(1)
            colormap('gray')
            
            subplot(6,6,i)
            imagesc(filter1)
            axis off
            V(:,:,i) = filter1;

            i = i + 1;
            subplot(6,6,i)
            imagesc(filter2)
            axis off
            V(:,:,i) = filter2;
        end
        
        if 1
            %figure(1)
            %colormap('gray')
            % Do convolution with image
            i = i + 1;
            result(:,:,i) = conv2(image, filter1,'valid');
            %subplot(6,6,i);
            %imagesc(result(:,:,i));
            %axis off
            i = i + 1;
            result(:,:,i) = conv2(image, filter2, 'valid');
            %subplot(6,6,i);
            %imagesc(result(:,:,i));
            %axis off
        end %convolution	
        
    end %scale
end %orientation
