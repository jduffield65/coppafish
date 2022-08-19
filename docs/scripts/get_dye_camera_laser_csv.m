%% get_dye_camera_laser_csv
% Script to produce csv file indicating expected intensity of every dye
% with each camera/laser combination.
%% Global data considering all dyes in all cameras/lasers
% spreadsheet contains info on intensity of dye in each laser/camera
% combination.
input_directory = '/Users/joshduffield/Documents/UCL/ISS/quad_cam/'; 
dye_info_file = [input_directory, 'dye_plots.xlsx'];
% dye names are names of sheets in excel files
sheet_names = sheetnames(dye_info_file);
dyes = sheet_names(~(sheet_names=='legend'));
[~,dye_sheet_no] = ismember(dyes,sheet_names);

% in legend sheet, get details of which cameras/lasers correspond to which
% index in dye spreadsheets
lasers = xlsread(dye_info_file, find(sheet_names == "legend"), 'e2:e8');
cameras = xlsread(dye_info_file, find(sheet_names == "legend"), 'b2:b5');

% gives the index in each spreadsheet of intensity of that dye
% with first laser and first camera.
% Camera indicated by column, laser indicated by row.
% use raw-background as approximate intensity of filtered images
letter2number = @(c)1+lower(c)-'a';
number2letter = @(n)char(n-1+'a');
start_intensity_column = letter2number('b');
start_intensity_row = 22;

n_dyes = size(dyes,1);
n_cameras = length(cameras);
n_lasers = length(lasers);

%% Read off intensities for each dye from global spreadsheet
n_data = n_dyes * n_cameras * n_lasers;
data_cell = cell(n_data+1, 4);
data_cell{1,1} = 'Dye';
data_cell{1,2} = 'Camera';
data_cell{1,3} = 'Laser';
data_cell{1,4} = 'Intensity';
i = 2;
for d=1:n_dyes
    for c=1:n_cameras
        for l=1:n_lasers
            data_cell{i,1} = dyes{d};
            data_cell{i,2} = cameras(c);
            data_cell{i,3} = lasers(l);
            col = number2letter(start_intensity_column+c-1);
            row = start_intensity_row + l-1;
            excel_cell = col+string(row);
            data_cell{i, 4} = xlsread(dye_info_file, dye_sheet_no(d), excel_cell);
            i = i+1;
        end
    end
end
T = cell2table(data_cell(2:end,:),'VariableNames',data_cell(1,:));
OutputDirectory =  '/Users/joshduffield/Desktop/';
writetable(T,fullfile(OutputDirectory,'dye_camera_laser_raw_intensity.csv'));