function [Iinterp,zinterp,vx,vy] = sliceInterp_spline_intensitySpline(I,z,R,lambda,tau,TOL,maxIter,borderSize,vx,vy)
% Curvature regularized, registration-based spline-spline slice interpolation.
%
% If you want to use this algorithm please cite:
%
% Antal Horvath, Simon Pezold, Matthias Weigel, Katrin Parmar, and Philippe Cattin
% "High Order Slice Interpolation for Medical Images."
% Simulation and Synthesis in Medical Imaging (MICCAI 2017 Workshop SASHIMI). 2017
%
% Input:
% I = 3D double array of size M x N x P; a list of matrizes of same dimensions; slices with similar objects
% z = 1D double array of size P; z positions of the parallel slices; important are the relative distances h between each other, see code
% R = double; refinement factor; ratio of new dz over old dz
%
% Optional Input:
% lambda = double; default = 10; curvature regularization factor
% tau = double; default = 10; implicit gradient descent step size
% TOL = double; default = 0.001; stopping criteria; tolerance of the max(L1(L2)) norm of the differencec of the last 10 updates of the displacement fields v
% maxIter = integer; default = 100; maximal amount of iterations
% borderSize = double; default = 0.1; relative border size w.r.t. M and N that is used to cut off the image during registration because of zero padding artifacts
% vx,vy = 3D double arrays of size MxNx(P-1); default = zeros(M,N,P-1); initial displacement field between neighbouring slices in x and y direction
%
% Output:
% Iinterp = 3D double array of size MxNx(R(P-1)+1); the stack of interpolated slices + the original ones
% zinterp = 1d double array of size (R(P-1)+1); z positions of the interpolated slices + the original ones
% vx,vy = 3D double arrays of size MxNx(P-1); registered displacement field between neighbouring slices in x and y direction

%% prepare
if ~exist('lambda','var')
    lambda = 10;
end
if ~exist('tau','var')
    tau = 10;
end
if ~exist('TOL','var')
    TOL = 0.001;
end
if ~exist('maxIter','var')
    maxIter = 100;
end
if ~exist('borderSize','var')
    borderSize = 0.1;
end
if ndims(I) ~= 3
    disp('I must be an array of matrices, i.e. ndims(I)==3.')
    return
end
[M, N, P] = size(I);
if ~exist('vx','var')
    vx = zeros(M, N, P-1);
end
if ~exist('vy','var')
    vy = zeros(M, N, P-1);
end

%% further parameters
sigma = 1; % variance of the gaussian derivative used for convolutional derivative calculation
paddingScheme = 'reflecting'; % check if 'reflecting' makes sense in your case; options are 'zero', 'reflecting', 'reflectingEven', 'constant' and 'periodic'
borderSizem = ceil(borderSize * M);
borderSizen = ceil(borderSize * N);
numberOfCollectedErrors = 10;
enablePlotting = false; % set to true if you want to spy on iterations
plotAfterAmountOfIterations = 10; % afer every plotAfterAmountOfIterations-th iteration it visualizes the current iteration if 'enablePlotting' is true.

%% initialization
% calc spatial image derivatives with convolutions of gaussian
% derivatives in x and y direction
[gx, gy] = derivativesOfGauss(sigma,sigma);
I_x = zeros(M, N, P);
I_y = I_x;
for k = 1:P
    I_x(:,:,k) = convolve(I(:,:,k), gx, paddingScheme);
    I_y(:,:,k) = convolve(I(:,:,k), gy, paddingScheme);
end

% calculate the spaces between the slices
h = diff(z);

% calc new z axis points
zinterp = [kron(z(1:end-1),ones(1,R)) + kron( h, (0:(R-1))/R ),z(end)];

% create spline matrix, its inverse and its cholesky decomposition
% the spline matrix corresponds to equation (3) in the paper
e = 1./h';
A = spdiags([[e(1:end-1);1;1] [2; 2*(e(1:end-1)+e(2:end));2] [1;1;e(2:end)]], -1:1, P, P );
Ainv = A^(-1);
A = (A + A') / 2; %%% I added this
L = chol(A,'lower');

% calculate the initial spline coefficients for the spline trajectories
[ax,ay] = calculateSplineCoefficientsForSplineTrajectories;

% calc grid for image in-plane interpolations
[Y,X] = ndgrid( 1:M, 1:N );

% set up the curvature regularization eigenvalues
d = -4 + 2 * kron( ones(1,N), cos(((1:M)-1)'*pi/M ) ) + 2 * kron( cos( ((1:N)-1)*pi/N ), ones(M,1) );
% modify for implicit gradient descent stepping
d = 1./(1 + tau * lambda * d.^2);

% error initialization
errors = zeros(numberOfCollectedErrors,1);
vx_new = zeros(M,N,P-1);
vy_new = vx_new;

%% registration iterations
for iteration = 1:maxIter
    %% update displacement fields between all slices
    for k = 2:P
        % transforming the image domains with the hermite interpolation
        % polynomials
        % corresponds to equation (3) in the paper
        v_x = vx(:,:,k-1);
        v_y = vy(:,:,k-1);
        TXdown = X - v_x/2 - (-1/8) * (ax(:,:,k) - ax(:,:,k-1)) * h(k-1);
        TYdown = Y - v_y/2 - (-1/8) * (ay(:,:,k) - ay(:,:,k-1)) * h(k-1);
        TXup = X + v_x/2 - (-1/8) * (ax(:,:,k) - ax(:,:,k-1)) * h(k-1);
        TYup = Y + v_y/2 - (-1/8) * (ay(:,:,k) - ay(:,:,k-1)) * h(k-1);
        
        % transforming and interpolating the images (with the spline
        % intensity interpolation equation (8) in the paper: the cubic
        % terms cancel out at position 0, 1/2 and 1.
        % force term at s = 1/2
        Idiff = interp2(I(:,:,k), TXup, TYup, 'linear', 0) - interp2(I(:,:,k-1), TXdown, TYdown, 'linear', 0);
        Iadd_x = (interp2(I_x(:,:,k-1), TXdown, TYdown, 'linear', 0) + interp2(I_x(:,:,k), TXup, TYup, 'linear', 0)) / 2;
        Iadd_y = (interp2(I_y(:,:,k-1), TXdown, TYdown, 'linear', 0) + interp2(I_y(:,:,k), TXup, TYup, 'linear', 0)) / 2;
        
        % force term at s = 0
        Idiff0 = interp2(I(:,:,k), X + v_x, Y + v_y, 'linear', 0) - I(:,:,k-1);
        I_x0 = interp2(I_x(:,:,k), X + v_x, Y + v_y, 'linear', 0);
        I_y0 = interp2(I_y(:,:,k), X + v_x, Y + v_y, 'linear', 0);
        
        % force term at s = 1
        Idiff1 = I(:,:,k) - interp2(I(:,:,k-1), X - v_x, Y - v_y, 'linear', 0);
        I_x1 = interp2(I_x(:,:,k-1), X - v_x, Y - v_y, 'linear', 0);
        I_y1 = interp2(I_y(:,:,k-1), X - v_x, Y - v_y, 'linear', 0);
        
        % add the forces
        F1 = (Idiff .* Iadd_x + Idiff1 .* I_x1 + Idiff0 .* I_x0) / 3;
        F2 = (Idiff .* Iadd_y + Idiff1 .* I_y1 + Idiff0 .* I_y0) / 3;
        
        %% iterate (through solving the linear system caused by the implicit stepping)
        vx_new(:,:,k-1) = idct2( dct2(v_x - tau * F1) .* d);
        vy_new(:,:,k-1) = idct2( dct2(v_y - tau * F2) .* d);
    end
    
    %% error calculation
    % calculate the L1 norm of the pointwise (vx,vy)-L2-norms
    e = elementwiseSum(sqrt((vx(1+borderSizem:end-borderSizem,1+borderSizen:end-borderSizen,:)-...
        vx_new(1+borderSizem:end-borderSizem,1+borderSizen:end-borderSizen,:)).^2 + ...
        + (vy(1+borderSizem:end-borderSizem,1+borderSizen:end-borderSizen,:)-...
        vy_new(1+borderSizem:end-borderSizem,1+borderSizen:end-borderSizen,:)).^2)) / ((M-2*borderSize)*(N-2*borderSize)*(P-1));
    
    % collect the errors
    addError(e)
    
    % get the largest error out of the last 10 errors
    error = max(errors);
    disp(['iteration ' num2str(iteration) ': error = ' num2str(error)])
    
    %% updating
    vx = vx_new;
    vy = vy_new;
    
    %% update spline coefficients for the spline trajectories
    [ax,ay] = calculateSplineCoefficientsForSplineTrajectories;
    
    %% plotting
    if enablePlotting && mod(iteration, plotAfterAmountOfIterations) == 1
        Iinterp = sliceInterpolate;
        myPlot
    end
    
    %% stopping criteria
    % check if stopping criteria is fulfilled
    if error < TOL
        disp(['Tolerance reached. ' num2str(iteration) ' iterations needed.'])
        break
    end
    
    % notify if maximum number of iterations were needed
    if iteration == maxIter
        disp('Maximum number of iterations needed.')
    end
end

Iinterp = sliceInterpolate;

%% nested functions
    function [ax,ay] = calculateSplineCoefficientsForSplineTrajectories
        % set up the inhomogeneities
        % corresponds to equation (3) in the paper
        prov_x = reshape( reshape(vx,[M,N*(P-1)])./kron(h,ones(M,N)) , [M,N,(P-1)]);
        prov_y = reshape( reshape(vy,[M,N*(P-1)])./kron(h,ones(M,N)) , [M,N,(P-1)]);
        d_x = 3 * (cat(3,zeros(M,N),prov_x) + cat(3,prov_x,zeros(M,N)));
        d_y = 3 * (cat(3,zeros(M,N),prov_y) + cat(3,prov_y,zeros(M,N)));
        
        % solve for the spline coefficients a
        % corresponds to equation (4) in the paper:
        % here we implemented a simpel tensor multiplication with the inverse
        % of A, instead of the forward and backward substitutions with the
        % cholesky decomposed triangular matrix L
        ax = permute(reshape(Ainv * reshape(permute(d_x,[3 1 2]),[P,M*N]),[P,M,N]),[2,3,1]);
        ay = permute(reshape(Ainv * reshape(permute(d_y,[3 1 2]),[P,M*N]),[P,M,N]),[2,3,1]);
    end

    function Iinterp = sliceInterpolate
        % intensity spline interpolation along the spline trajectories
        % set up the intensity spline inhomogeneities
        % corresponds to equation (6) in the paper
        dI = zeros(M, N, P);
        
        % transform the domain
        TXl = X + vx(:,:,1);
        TYl = Y + vy(:,:,1);
        % interpolate the subsequent slices for the corresponding z position
        dI(:,:,1) = 3 * ( interp2(I(:,:,2), TXl, TYl, 'cubic', 0) - I(:,:,1) ) / h(1);
        
        for ii = 2:P-1
            % transform the domain
            TXl = X + vx(:,:,ii);
            TYl = Y + vy(:,:,ii);
            TXr = X - vx(:,:,ii-1);
            TYr = Y - vy(:,:,ii-1);
            % interpolate the subsequent slices for the corresponding z position
            dI(:,:,ii) = 3 * ( (I(:,:,ii) - interp2(I(:,:,ii-1), TXr, TYr, 'cubic', 0)) / h(ii-1) + (interp2(I(:,:,ii+1), TXl, TYl, 'cubic', 0) - I(:,:,ii)) / h(ii) );
        end
        
        % transform the domain
        TXr = X - vx(:,:,P-1);
        TYr = Y - vy(:,:,P-1);
        % interpolate the subsequent slices for the corresponding z position
        dI(:,:,P) = 3 * ( I(:,:,P) - interp2(I(:,:,P-1), TXr, TYr, 'cubic', 0) ) / h(P-1);
        
        % solve for intensity spline coefficients
        % corresponds to equation (7) in the paper
        intermediateResult = forwardSubstitution(L, dI);
        aI = backwardSubstitution(L', intermediateResult);
        
        % slice intensity spline interpolation
        Iinterp = zeros(M,N, (P-1)*R + 1);
        Iinterp(:,:,1) = I(:,:,1);
        l = 2;
        for ii = 2:P
            I1 = I(:,:,ii-1);
            I2 = I(:,:,ii);
            currentv_x = vx(:,:,ii-1);
            currentv_y = vy(:,:,ii-1);
            for jj = 1:R-1
                % transform the domain
                s = jj / R;
                [TXm, TYm] = hermiteIP(X, Y, currentv_x, currentv_y, s, ii, 'right');
                [TXp, TYp] = hermiteIP(X, Y, currentv_x, currentv_y, 1-s, ii, 'left');
                
                % interpolate the subsequent sline coefficients for the corresponding z position
                % corresponds to equation (8) in the paper
                I1interp = interp2(I1, TXm, TYm, 'cubic', 0);
                delta = interp2(I2, TXp, TYp, 'cubic', 0) - I1interp;
                Iinterp(:,:,l) = I1interp + s * delta + ...
                    + s*(s-1)*( s * ( interp2(aI(:,:,ii), TXp, TYp, 'cubic', 0) * h(ii-1) - delta) + (s-1) * ( interp2(aI(:,:,ii-1), TXm, TYm, 'cubic', 0) * h(ii-1) - delta) );
                l = l+1;
            end
            Iinterp(:,:,l) = I2;
            l = l+1;
        end
        % Iinterp = Iinterp(1 + borderSizem:end - borderSizem,1 + borderSizen:end - borderSizen,:);
        
        %% nested nested functions
        function x = forwardSubstitution(A,b)
            x = zeros(M,N,P);
            x(:,:,1) = b(:,:,1) / A(1,1);
            for iii = 2:P
                TXr = X - vx(:,:,iii-1);
                TYr = Y - vy(:,:,iii-1);
                x(:,:,iii) = (b(:,:,iii) - A(iii,iii-1) * interp2(x(:,:,iii-1), TXr, TYr, 'cubic', 0) ) / A(iii,iii);
            end
        end
        
        function x = backwardSubstitution(A,b)
            x = zeros(M,N,P);
            x(:,:,P) = b(:,:,P) / A(P,P);
            for iii = (P-1):-1:1
                TXl = X + vx(:,:,iii);
                TYl = Y + vy(:,:,iii);
                x(:,:,iii) = (b(:,:,iii) - A(iii,iii+1) * interp2(x(:,:,iii+1), TXl, TYl, 'cubic', 0) ) / A(iii,iii);
            end
        end
        
        function [x,y] = hermiteIP(x,y,vx,vy,s,i,type)
            if strcmp(type,'right')
                x = x - s * vx - s * (s-1) * ( s * (ax(:,:,i) * h(i-1) - vx) + (s-1) * (ax(:,:,i-1) * h(i-1) - vx) );
                y = y - s * vy - s * (s-1) * ( s * (ay(:,:,i) * h(i-1) - vy) + (s-1) * (ay(:,:,i-1) * h(i-1) - vy) );
            elseif strcmp(type,'left')
                x = x + s * vx - s * (s-1) * ( s * (ax(:,:,i) * h(i-1) - vx) + (s-1) * (ax(:,:,i-1) * h(i-1) - vx) );
                y = y + s * vy - s * (s-1) * ( s * (ay(:,:,i) * h(i-1) - vy) + (s-1) * (ay(:,:,i-1) * h(i-1) - vy) );
            else
                disp('Unknown hermite IP type.')
            end
        end
    end

    function myPlot
        figure(2);clf
        l = 1;
        fl = 1;
        for dummy1 = 1:P-1
            for dummy2 = 1:R
                subplot(P-1,R+1,fl)
                imshow(Iinterp(:,:,l), [0,1])
                l = l+1;
                fl = fl+1;
            end
            subplot(P-1,R+1,fl); fl = fl+1;
            imshow(Iinterp(:,:,l), [0,1])
        end
        disp(['errors: ' num2str(error)])
        % figure(3);clf
        % for i = 1:size(vx,3)
        %     subplot(1,size(vx,3),i)
        %     imshow(vx(:,:,i),[])
        % end
        pause(0.01)
    end

    function addError(e)
        errors(2:end) = errors(1:end-1);
        errors(1) = e;
    end
end

%% external functions
function [gx, gy] = derivativesOfGauss(sigmax,sigmay)
% Calculates a 2D gauss kernel (respectively its derivatives in x and y direction
% for mu = 0 and variance (sigmax,sigmay).
% The size of the kernel is calculated automatically, such that the
% area under the kernel is more than 99% of the analytically weighted domain.
%
% Input:
% sigmax = double; gaussian variance in x direction
% sigmay = double; gaussian variance in y direction
%
% Output:
% gx = double matrix; derivative in x direction of the 2D gaussian
% gy = double matrix; derivative in y direction of the 2D gaussian

% calculate the size n of the kernel such that it covers more than
% 99% of the area under the gaussian.
nx = ceil(3 * sigmax);
ny = ceil(3 * sigmay);

% gaussian in x
x = -nx : nx;
gaussx = 1 / (sqrt(2*pi) * sigmax) * exp( -(x/sigmax).^2 / 2 );

% gaussian in y
y = -ny : ny;
gaussy = 1 / (sqrt(2*pi) * sigmay) * exp( -(y/sigmay).^2 / 2 );

% derivative in x direction of gaussian in x
delxgaussx = -gaussx .* x / sigmax^2;

% derivative in y direction of gaussian in y
delygaussy = -gaussy .* y / sigmay^2;

% derivative in x direction of the 2D gaussian
gx = gaussy' * delxgaussx; %outer product

% derivative in y direction of the 2D gaussian
gy = delygaussy' * gaussx; %outer product
end

function w = convolve(u,k,paddingScheme)
% Convolves a 2D image with a 2D kernel in the Fourier domain.
%
% Input:
% u = matrix; 2D image to be convolved with the kernel k
% k = matrix of a smaller size than the size of u; the sizes in x and y direction have to be odd; 2D kernel
% paddingScheme = a string in {'zero','reflecting','reflectingEven','constant','periodic'}
%
% Output:
% w = matrix of same size than u; convolved 2D image

%% calculate sizes
[m,n] = size(u);
[o,p] = size(k);
oh = floor(o/2);
ph = floor(p/2); % half of the kernel sizes

%% kernel padding
kp = zeros(m+o-1,n+p-1);
kp(1:o,1:p) = k;

%% image padding
if strcmp(paddingScheme,'zero')
    up = zeros(m+o-1,n+p-1);
    up(1:m,1:n) = u;
elseif strcmp(paddingScheme,'reflecting')
    % a b c b a
    u = [ u(oh+1:-1:2,:); u; u(end-1:-1:end-oh,:) ];
    up = [ u(:,ph+1:-1:2), u, u(:,end-1:-1:end-ph) ];
elseif strcmp(paddingScheme,'reflectingEven')
    % a b c c b a
    u = [ u(oh:-1:1,:); u; u(end:-1:end-oh+1,:) ];
    up = [ u(:,ph:-1:1), u, u(:,end:-1:end-ph+1) ];
elseif strcmp(paddingScheme,'constant')
    u = [ kron(ones(oh,1),u(1,:)); u; kron(ones(oh,1),u(end,:)) ];
    up = [ kron(ones(1,ph),u(:,1)), u, kron(ones(1,ph),u(:,end)) ];
elseif strcmp(paddingScheme,'periodic')
    u = [ u( end-oh+1:end , :); u; u( 1:oh, :) ];
    up = [ u(:, end-ph+1:end ), u, u(:, 1:ph ) ];
else
    disp('Define the padding scheme for the convolution.')
    w = u;
    return
end

%% convolution
w = ifft2(fft2(up) .* fft2(kp));

%% taking the correct part of the picture because of translation of the convolution
if strcmp(paddingScheme,'zero')
    w = w(1+oh:m+oh,1+ph:n+ph);
else
    w = w(1+2*oh:m+2*oh, 1+2*ph:n+2*ph);
end
end

function value = elementwiseSum(A)
value = sum(A(:));
end
