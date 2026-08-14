import agama
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .mixture import AutoGaussianMixtureModel, util, preprocessing
from .PyTNG.snapshot_loader import Snapshot
from .gravity.kinematic_solver import construct_galaxy_potential_model, calculate_kinematic_param
from .visualize import visualize_decomposition
from .config import BASEPATH, check_basepath

RCUT_RANGE = [1, 7]

def train_auto_gaussian_mixture_model(galaxy, pot, jzojc_cut=0.5):

    eoemin_index = 0
    jzojc_index = 1
    jpojc_index = 2
    X = np.column_stack([galaxy.s['eoemin'], galaxy.s['jzojc'], galaxy.s['jpojc']])
    keep_particle = (galaxy.s['eoemin']<0)&(np.abs(galaxy.s['jzojc'])<1.5)&(galaxy.s['jpojc']<1.5)
    sph, _ = util.JEHistogram(galaxy.s['eoemin'][keep_particle], galaxy.s['jzojc'][keep_particle], n_E=25, n_eps=50)
    sph = (sph) & (np.abs(galaxy.s['jzojc'][keep_particle])<=0.5)
    eoemin_cut = util.get_Ecut(galaxy.s['eoemin'][keep_particle][sph],
                               galaxy.s['mass'][keep_particle][sph],
                               M_bin=100, m_bin=25, Mmin=0.1)
    r = np.logspace(-1, 1, 100)
    points = np.column_stack((r*0, r*0, r))
    potential = pot.potential(points)
    max_eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, RCUT_RANGE[1])]
    min_eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, RCUT_RANGE[0])]
    if eoemin_cut == 0 or max_eoemin_cut < eoemin_cut or eoemin_cut < min_eoemin_cut:
        eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, 3.5)]

    scaler = preprocessing.RobustScaler()
    X_train= scaler.fit_transform(X[keep_particle])

    eoemin_cut_train = scaler.transform(eoemin_cut, columns=eoemin_index)
    jzojc_cut_train = scaler.transform(jzojc_cut, columns=jzojc_index)
    r_jzojc_cut_train = scaler.transform(-jzojc_cut, columns=jzojc_index)

    auto_gmm = AutoGaussianMixtureModel(random_state=42)
    auto_gmm = auto_gmm.fit(X_train, 
                            eoemin_cut=eoemin_cut_train, 
                            jzojc_cut=jzojc_cut_train,
                            r_jzojc_cut = r_jzojc_cut_train, 
                            max_iter=100, 
                            min_iter=10,
                            scaler=scaler,
                            use_float32=True)

    best_model = scaler.inverse_transform_GMM(auto_gmm.best_model)
    return X, best_model, eoemin_cut, jzojc_cut
  
def kinematic_decomposition_pipeline(run, snapNum, subID, 
                                     gravity_potential_path=None, 
                                     image_path=None, 
                                     structure_properties_output_path=None,
                                     mixture_model_output_path=None):

    check_basepath()
    basePath = f"{BASEPATH}/{run}/output"

    if gravity_potential_path is not None:
        filename = f"{gravity_potential_path}/{subID}.ini"
        if not Path(filename).exists():
            print("create multipole potential ...")
            snapshot = Snapshot(basePath, snapNum)
            load_particle_fields = 'potential'
            snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
            snapshot.physical_units()
            snapshot.load_group_catalog(ID=subID)
            snapshot.GC_physical_units()
            snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
            snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
            galaxy = snapshot.container
            pot = construct_galaxy_potential_model(galaxy)
            pot.export(filename) 
        pot = agama.Potential(filename)

    snapshot = Snapshot(basePath, snapNum)
    load_particle_fields = {"star": ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'GFM_StellarFormationTime'],
                            "dm": ['Coordinates', 'Velocities', 'Masses'],
                            "gas": ['Coordinates', 'Velocities', 'Masses']}
    snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=subID)
    snapshot.GC_physical_units()
    snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
    snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
    galaxy = snapshot.container
    if gravity_potential_path is None:
        pot = construct_galaxy_potential_model(galaxy)
    galaxy = calculate_kinematic_param(galaxy, pot)
    X, model, eoemin_cut, jzojc_cut = train_auto_gaussian_mixture_model(galaxy, pot)
    galaxy     = util.decompose(X, galaxy, model, eoemin_cut, jzojc_cut, predict_method='hard')
    if mixture_model_output_path is not None:
        mixture_model_output = util.decompose_mixture_model(model, eoemin_cut, jzojc_cut, -jzojc_cut)
        with open(f"{mixture_model_output_path}/mixture_model_{run}_{snapNum}_{subID}.pkl", "wb") as f:
            pickle.dump(mixture_model_output, f)
    if image_path is not None:
        visualize_decomposition(X, model, galaxy, eoemin_cut, jzojc_cut, threshold_line=True, ranges=None)
        plt.savefig(Path(image_path)/f"{subID}.pdf", dpi=300, bbox_inches='tight')
    if structure_properties_output_path is not None:
        structure_properties_output = util.save_structure_properties(galaxy)
        with open(f"{structure_properties_output_path}/structure_properties_{run}_{snapNum}_{subID}.pkl", "wb") as f:
            pickle.dump(structure_properties_output, f)
    return model, galaxy, eoemin_cut, jzojc_cut

 
