from kinematic_decompose.config import TEST_IMAGE_PATH, TEST_DATA_PATH
from kinematic_decompose.pipeline import kinematic_decomposition_pipeline

run = "TNG50-1"
snapNum = 99 
subID = 307486
model, galaxy, eoemin_cut, jzojc_cut = kinematic_decomposition_pipeline(run="TNG50-1", snapNum=snapNum, subID=subID,
                                                                        gravity_potential_path=TEST_DATA_PATH,
                                                                        image_path=TEST_IMAGE_PATH,
                                                                        structure_properties_output_path=TEST_DATA_PATH,
                                                                        mixture_model_output_path=TEST_DATA_PATH)
import pickle
with open(f"{TEST_DATA_PATH}/structure_properties_{run}_{snapNum}_{subID}.pkl", "rb") as f:
    data = pickle.load(f)
for main_key, sub_dict in data.items():
    print(f"\n{main_key}:")
    for sub_key, value in sub_dict.items():
        print(f"  {sub_key}: {value}")

