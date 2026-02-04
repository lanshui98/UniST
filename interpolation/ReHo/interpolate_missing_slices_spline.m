function interpolate_missing_slices_spline(inDir, outDir, varargin)
% Interpolate missing slices between adjacent existing slices using
% sliceInterp_spline_intensitySpline for spline/intensity interpolation,
% generating missing slice_XXX.tif files.
%
% Requires: sliceInterp_spline_intensitySpline must be callable in the path.
%
% R setting: For each pair of endpoints i0,i1, automatically set R = (i1 - i0 + 1),
% so that the returned slices_interpolated indices cover i0..i1 (including endpoints),
% and we only save the missing intermediate slices i0+1..i1-1.

p = inputParser;
addParameter(p, 'CopyOriginal', true, @islogical); % Whether to copy original slices to outDir
addParameter(p, 'Binarize', false, @islogical);
addParameter(p, 'Thresh', 127, @(x)isnumeric(x)&&isscalar(x));
% sliceInterp parameters
addParameter(p, 'Lambda', 10, @(x)isnumeric(x)&&isscalar(x));
addParameter(p, 'Tau', 100, @(x)isnumeric(x)&&isscalar(x));
addParameter(p, 'TOL', 0.01, @(x)isnumeric(x)&&isscalar(x));
addParameter(p, 'MaxIter', 1000, @(x)isnumeric(x)&&isscalar(x));
addParameter(p, 'BorderSize', 0.1, @(x)isnumeric(x)&&isscalar(x));
parse(p, varargin{:});

copyOriginal = p.Results.CopyOriginal;
doBin = p.Results.Binarize;
thresh = p.Results.Thresh;
lambda = p.Results.Lambda;
tau = p.Results.Tau;
TOL = p.Results.TOL;
maxIter= p.Results.MaxIter;
borderSize = p.Results.BorderSize;

if ~exist(outDir,'dir'); mkdir(outDir); end

% Collect files and indices
listing = dir(fullfile(inDir,'slice_*.tif'));
if isempty(listing), error('No slice_XXX.tif files found in directory: %s', inDir); end

idx = zeros(numel(listing),1);
for i=1:numel(listing)
    tok = regexp(listing(i).name,'slice_(\d+)\.tif$','tokens','once');
    if isempty(tok), error('Filename does not match pattern: %s', listing(i).name); end
    idx(i) = str2double(tok{1});
end
[sortedIdx, ord] = sort(idx);
files = {listing(ord).name};

% Read one image to determine dimensions and type
probe = imread(fullfile(inDir, files{1}));
if ndims(probe)==3, probe_gray = mean(im2single(probe),3); else, probe_gray = im2single(probe); end
[M,N] = size(probe_gray);

% Option: Copy existing slices to output directory first
if copyOriginal
    for i=1:numel(files)
        copyfile(fullfile(inDir, files{i}), fullfile(outDir, files{i}));
    end
end

% Interpolate between each pair of adjacent endpoints
for k = 1:numel(sortedIdx)-1
    i0 = sortedIdx(k);
    i1 = sortedIdx(k+1);
    gap = i1 - i0;
    if gap <= 1
        continue; % Consecutive, no interpolation needed
    end
    
    % Read both endpoint slices and convert to single precision grayscale
    I0 = read_gray_single(fullfile(inDir, sprintf('slice_%03d.tif', i0)));
    I1 = read_gray_single(fullfile(inDir, sprintf('slice_%03d.tif', i1)));
    assert(isequal(size(I0),[M,N]) && isequal(size(I1),[M,N]), 'Size mismatch');
    
    % Assemble 3D volume: P=2
    threeDArray = zeros(M,N,2,'single');
    threeDArray(:,:,1) = I0;
    threeDArray(:,:,2) = I1;
    
    % z takes actual indices, R takes total number of slices from i0..i1
    z = [i0, i1];
    R = i1 - i0 + 1;
    
    % Call interpolation function (returned slice indices should cover i0..i1)
    [slices_interpolated, z_interpolated, vx, vy] = ...
        sliceInterp_spline_intensitySpline( ...
        threeDArray, z, R, lambda, tau, TOL, maxIter, borderSize);
    
    % Save missing intermediate slices
    for miss = i0+1 : i1-1
        % Find position of miss in z_interpolated
        pos = find(z_interpolated==miss, 1);
        if isempty(pos)
            % Some implementations return equidistant indices 1..R, 
            % fall back to linear proportional mapping
            pos = round( (miss - i0)/(i1 - i0) * (R-1) ) + 1;
            pos = max(1, min(R, pos));
        end
        
        J = slices_interpolated(:,:,pos); % single, 0..1 or original magnitude
        
        % Optional binarization
        if doBin
            if isfloat(J), BW = J > (thresh/255); else, BW = J > thresh; end
            Jout = uint8(BW) * 255;
        else
            % Keep as 8-bit grayscale
            if isfloat(J)
                Jout = im2uint8(mat2gray(J)); % Normalize for safety
            else
                Jout = uint8(J);
            end
        end
        
        imwrite(Jout, fullfile(outDir, sprintf('slice_%03d.tif', miss)));
    end
end

fprintf('Completed: Missing slices written to %s\n', outDir);
end

function I = read_gray_single(path)
I0 = imread(path);
if ndims(I0)==3
    I = mean(im2single(I0),3);
else
    I = im2single(I0);
end
end
