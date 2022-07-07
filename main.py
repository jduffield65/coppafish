from iss import run_pipeline
from iss.utils.export_to_pciseq import export

# ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/B8S5_Slice001/start_params.ini'
# ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d_full/anne_3d_full.ini'
notebook = run_pipeline(ini_file)

# Export spot coordinates and decoded gene to pciSeq
export(config_file_path=ini_file)
