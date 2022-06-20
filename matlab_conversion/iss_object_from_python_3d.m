function o = iss_object_from_python_3d(python_file_name, method)
%Loads in nb info outputted by python and adds it to matlab o object so it
%can be plotted.
o = iss; %use iss-QuadCam3D branch

% Set variables from Python File
load(python_file_name)
o.BledCodes = permute(bled_codes_ge, [1,3,2]); % from r,c to c,r
o.nRounds = size(o.BledCodes, 3);
o.nBP = size(o.BledCodes, 2);
o.UseChannels = find(~isnan(squeeze(bled_codes(1,1,:))))';
o.UseRounds = find(~isnan(squeeze(bled_codes(1,:,1))));
o.bpLabels = cellstr(num2str((0:o.nBP-1)'))';
nCodes = size(o.BledCodes,1);
% gene_efficiency bled codes are used for both dot product and omp method.
o.BledCodes = o.BledCodes(:,:);
o.NormBledCodes = o.BledCodes;
o.BleedMatrix = permute(bleed_matrix, [2, 3, 1]); % from r,c,d to c,d,r.
o.CharCodes = cell(nCodes,1);
o.GeneNames = cell(nCodes,1);
for g=1:nCodes
    o.CharCodes{g} = sprintf('%0d', gene_codes(g,:));
    o.GeneNames{g} = strrep(convertCharsToStrings(gene_names(g,:)),' ','');
    o.GeneNames{g} = convertStringsToChars(o.GeneNames{g});
end
if strcmpi(method, 'omp')
    o.cSpotColors = permute(omp_colors, [1,3,2]); % from r,c to c,r
    o.SpotCodeNo = transpose(omp_gene_no) + 1; % matlab indexing starts at 1 not 0.
    o.SpotIntensity =  transpose(omp_intensity);
    o.CombiIntensityThresh = omp_intensity_thresh;
    LocalTile = omp_tile + 1; % matlab indexing starts at 1 not 0.
    o.SpotGlobalYXZ = double(omp_local_yxz) + tile_origin(LocalTile,:) + 1;
    max_score = omp_score_multiplier * sum(omp_spot_shape(:)==1) + sum(omp_spot_shape(:)==-1);
    o.SpotScore = double(omp_score_multiplier * transpose(omp_n_neighbours_pos) + ...
        transpose(omp_n_neighbours_neg)) / max_score;
    o.CombiQualThresh = omp_score_thresh;
elseif strcmpi(method, 'dot_product')
    o.cSpotColors = permute(ref_spots_colors, [1,3,2]);
    o.SpotCodeNo = transpose(ref_spots_gene_no) + 1;
    o.SpotIntensity = transpose(ref_spots_intensity);
    o.CombiIntensityThresh = ref_spots_intensity_thresh;
    LocalTile = ref_spots_tile + 1; % matlab indexing starts at 1 not 0.
    o.SpotGlobalYXZ = double(ref_spots_local_yxz) + tile_origin(LocalTile,:) + 1;
    o.SpotScore = transpose(ref_spots_score);
    o.CombiQualThresh = ref_spots_score_thresh;
    o.cSpotIsolated = transpose(ref_spots_isolated);
else
    error("method must be 'omp' or 'dot_product'.")
end
o.SpotCombi = true(size(o.SpotScore));
o.SpotScoreDev = ones(size(o.SpotScore));
o.CombiDevThresh = 0;

o.BledCodesPercentile = permute(color_norm_factor, [2, 1]);  % from r,c to c,r
o.BledCodesPercentile = reshape(o.BledCodesPercentile, [1, o.nBP, o.nRounds]);
o.cNormSpotColors = bsxfun(@rdivide, double(o.cSpotColors), o.BledCodesPercentile);
nTiles = size(tile_origin, 1);
o.nExtraRounds=1;
o.ReferenceRound=8;
o.nBP = 7;
o.TileOrigin = zeros(nTiles,3,o.nRounds+o.nExtraRounds);
o.TileOrigin(:,:,o.ReferenceRound) =  tile_origin+1; % matlab indexing starts at 1 not 0.
o.A = permute(transform,[4,5,1,2,3]); % from t,r,c,T1,T2 to T1,T2,t,r,c
o.TileFiles = cell(o.nRounds+o.nExtraRounds, nTiles, 1, o.nBP);
for r=1:o.nRounds+o.nExtraRounds
    for t=1:nTiles
        for c=1:o.nBP
            o.TileFiles{r,t,1,c} = transpose(squeeze(tile_file_names(t,r,c,:)));
            o.TileFiles{r,t,1,c} = strrep(o.TileFiles{r,t,1,c},' ','');
        end
    end
end

end

