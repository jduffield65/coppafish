function o = iss_object_from_python_2d(python_file_name)
%Loads in nb info outputted by python and adds it to matlab o object so it
%can be plotted.
o = iss_OMP; %use iss-Josh branch

% Set variables not in Python
o.dpMethod = "WholeCode";
o.BackgroundWeightPower = 1; % (Not used in Python)
o.dpRemoveBackground = 1;
o.dpGeneEfficiencyMaxIter = 10;
o.ScaledKMeansFunc = 2;
o.GeneEfficiencyMinSpots = 30;
o.dpGeneEfficiencyPostBackground=true;
o.BleedMatrixPostBackground=true;

o.ompRemoveBackground = 1;
o.ompAnnulusNegativeCoef = false;
o.ompNoOverlapDotProduct = false;
o.ompIgnoreBadRound = false;
o.ompMethod = "Variance";
o.ompUseGeneEfficiency = true;
o.ompStopMethod = 'DotProduct';
o.ompIgnoreBackgroundBestGene = true;
o.ompAddNearIntense = false;
o.ompGeneImSmoothSz = 0;
o.ompNormBledCodesWeightDotProduct = true;
o.ompVarWeightNormCorrect = true;
o.ompInitialIntensityMethod = "Abs";
o.GeneRoundWeightMax=Inf;

% Set variables from Python File
load(python_file_name)
o.BledCodes = permute(bled_codes, [1,3,2]); % from r,c to c,r
o.nRounds = size(o.BledCodes, 3);
o.nBP = size(o.BledCodes, 2);
nCodes = size(o.BledCodes,1);
o.BledCodes = o.BledCodes(:,:);
o.BleedMatrix = permute(bleed_matrix, [2, 3, 1]); % from r,c,d to c,d,r.
o.BledCodesPercentile = permute(color_norm_factor, [2, 1]);  % from r,c to c,r
o.BledCodesPercentile = reshape(o.BledCodesPercentile, [1, o.nBP, o.nRounds]);
o.CharCodes = cell(nCodes,1);
o.GeneNames = cell(nCodes,1);
for g=1:nCodes
    o.CharCodes{g} = sprintf('%0d', gene_codes(g,:));
    o.GeneNames{g} = strrep(convertCharsToStrings(gene_names(g,:)),' ','');
    o.GeneNames{g} = convertStringsToChars(o.GeneNames{g});
end
o.GeneEfficiency = gene_efficiency;
o.GeneRoundWeight = gene_efficiency;
o.InitialBleedMatrix = permute(initial_bleed_matrix, [2, 3, 1]); % from r,c,d to c,d,r.
% Only keep one round
o.InitialBleedMatrix = o.InitialBleedMatrix(:, :, 1) .* squeeze(o.BledCodesPercentile);
% Background vectors don't have norm 1 in MATLAB but larger norm hence
% coefs need to be made smaller.
matlab_background_norm_factor = vecnorm(ones(o.nRounds,1),2,1);
o.ompBackgroundCoef = omp_background_coef / matlab_background_norm_factor;
o.dpBackgroundCoef = ref_spots_background_coef / matlab_background_norm_factor;
o.ompCoefs = omp_coef;
o.ompSpotColors = permute(omp_colors, [1,3,2]); % from r,c to c,r
o.dpSpotColors = permute(ref_spots_colors, [1,3,2]);
o.ompSpotCodeNo = transpose(omp_gene_no) + 1; % matlab indexing starts at 1 not 0.
o.dpSpotCodeNo = transpose(ref_spots_gene_no) + 1;
o.ompSpotIntensity = transpose(omp_intensity);
o.dpSpotIntensity = transpose(ref_spots_intensity);
o.ompIntensityThresh = omp_intensity_thresh;
o.dpIntensityThresh = ref_spots_intensity_thresh;
o.ompLocalTile = omp_tile + 1; % matlab indexing starts at 1 not 0.
o.dpLocalTile = ref_spots_tile + 1;
% Global got by adding tile origin to local. Then add 1 so smallest
% coordinate is (1,1) not (0,0).
o.ompSpotGlobalYX = double(omp_local_yxz(:, 1:2)) + tile_origin(o.ompLocalTile,1:2) + 1;
o.dpSpotGlobalYX = double(ref_spots_local_yxz(:, 1:2)) + tile_origin(o.dpLocalTile,1:2) + 1;
o.ompSpotScore = transpose([omp_n_neighbours_pos; omp_n_neighbours_neg]);
o.dpSpotScore = transpose(ref_spots_score);
SortCoefs=sort(o.ompCoefs,2,'descend');
CoefInd = sub2ind(size(o.ompCoefs), (1:size(o.ompCoefs,1))', o.ompSpotCodeNo);
o.ompSpotScoreDiff = o.ompCoefs(CoefInd) - SortCoefs(:,2);
o.dpSpotScoreDiff = transpose(ref_spots_score_diff);
o.ompScoreMultiplier = omp_score_multiplier;
o.ompScoreThresh = omp_score_thresh;
o.ompScoreDiffThresh = omp_score_thresh; % Only one omp_score_thresh in Python.
o.dpScoreThresh = ref_spots_score_thresh;
o.ompSpatialKernel = omp_spot_shape;
o.dpSpotIsolated = transpose(ref_spots_isolated);

nTiles = size(tile_origin, 1);
o.nExtraRounds=1;
o.AnchorRound=8;
o.ReferenceRound=o.AnchorRound;
o.TileOrigin = zeros(nTiles,2,o.nRounds+o.nExtraRounds);
o.TileOrigin(:,:,o.ReferenceRound) =  tile_origin(:,1:2)+1; % matlab indexing starts at 1 not 0.
o.D = permute(transform,[4,5,1,2,3]); % from t,r,c,T1,T2 to T1,T2,t,r,c
o.D = o.D([1,2,4],[1,2],:,:,:); % from 3d to 2d
o.TileFiles = cell(o.nRounds+o.nExtraRounds, nTiles);
for r=1:o.nRounds+o.nExtraRounds
    for t=1:nTiles
        o.TileFiles{r,t} = transpose(squeeze(tile_file_names(t,r,:)));
        o.TileFiles{r,t} = strrep(o.TileFiles{r,t},' ','');
    end
end

end

