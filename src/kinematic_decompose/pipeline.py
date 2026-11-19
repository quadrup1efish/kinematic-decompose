import pickle
import warnings
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path

from .potential import Potential
from .mixture import AutoGaussianMixtureModel, util, preprocessing
from .PyTNG.snapshot_loader import Snapshot
from .gravity.kinematic_solver import construct_galaxy_potential_model, calculate_kinematic_param

from .config import BASEPATH, check_basepath

DEFAULT_GMM_MAX_ITER = 500
DEFAULT_GMM_MIN_ITER = 50

def train_auto_gaussian_mixture_model(galaxy, pot, jzojc_cut=0.5):

    eoemin_index = 0
    jzojc_index = 1
    jpojc_index = 2
    X = np.column_stack([galaxy.s['eoemin'], galaxy.s['jzojc'], galaxy.s['jpojc']])
    keep_particle = (galaxy.s['eoemin']<0)&(np.abs(galaxy.s['jzojc'])<1.5)&(galaxy.s['jpojc']<1.5)
    sph, _ = util.JEHistogram(galaxy.s['eoemin'][keep_particle], galaxy.s['jzojc'][keep_particle], n_E=25, n_eps=50)
    sph = (sph) & (np.abs(galaxy.s['jzojc'][keep_particle])<=0.5)
    eoemin_cut = util.get_Ecut_skewt(
        galaxy.s['eoemin'][keep_particle][sph],
        galaxy.s['mass'][keep_particle][sph],
    )
    r_min = 2.0 * float(galaxy.properties['eps'])
    r_max = 10.0
    if not 0.0 < r_min < r_max:
        raise ValueError(f"Invalid energy-cut radius range [{r_min}, {r_max}] kpc")
    r = np.logspace(np.log10(r_min), np.log10(r_max), 100)
    points = np.column_stack((r*0, r*0, r))
    potential = pot.potential(points)
    potential_eoemin = potential / np.abs(galaxy.s['e'].min())
    #min_eoemin_cut = potential_eoemin[0]
    #max_eoemin_cut = potential_eoemin[-1]
    if (eoemin_cut is None or eoemin_cut == 0):# or max_eoemin_cut < eoemin_cut or eoemin_cut < min_eoemin_cut):
        fallback_radius = float(np.clip(1, r_min, r_max))
        eoemin_cut = potential_eoemin[np.searchsorted(r, fallback_radius)]

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
                            max_iter=DEFAULT_GMM_MAX_ITER,
                            min_iter=DEFAULT_GMM_MIN_ITER,
                            scaler=scaler,
                            use_float32=True)

    best_model = scaler.inverse_transform_GMM(auto_gmm.best_model)
    return X, best_model, eoemin_cut, jzojc_cut
  
def decompose_prepared_galaxy(galaxy, *, potential=None, trainer=None):
    """Decompose a centered, face-on galaxy without plotting or writing files.

    A custom trainer receives (galaxy, potential) and returns features, model,
    energy cut and circularity cut. Return those results plus galaxy and potential.
    """
    if potential is None:
        potential = construct_galaxy_potential_model(galaxy)
    galaxy = calculate_kinematic_param(galaxy, potential)
    if trainer is None:
        trainer = train_auto_gaussian_mixture_model
    X, model, eoemin_cut, jzojc_cut = trainer(galaxy, potential)
    galaxy = util.decompose(X, galaxy, model, eoemin_cut, jzojc_cut, predict_method='hard')
    return X, model, galaxy, eoemin_cut, jzojc_cut, potential


def kinematic_decomposition_pipeline(run, snapNum, subID, 
                                     gravity_potential_path=None, 
                                     image_path=None, 
                                     structure_properties_output_path=None,
                                     mixture_model_output_path=None, *,
                                     force_rebuild_potential=False):
    """Decompose a galaxy, optionally caching its potential under root/run/snap/ID.ini.

    Snapshot directories use three digits (e.g. 099). A None cache root disables
    disk caching. Legacy flat caches are never reused because their run and
    snapshot are unknown. Use force_rebuild_potential after changing construction
    settings; this cache intentionally has no sidecar metadata or automatic
    parameter invalidation. Completed exports replace the cache atomically.
    """

    check_basepath()
    basePath = f"{BASEPATH}/{run}/output"

    pot = None
    if gravity_potential_path is not None:
        root = Path(gravity_potential_path)
        filename = root / run / f"{snapNum:03d}" / f"{subID}.ini"
        if force_rebuild_potential or not filename.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
            legacy = root / f"{subID}.ini"
            if not filename.exists() and legacy.exists():
                warnings.warn(
                    f"Ignoring legacy potential cache {legacy}: its run and snapshot "
                    f"are unknown. Rebuilding {filename}; the legacy file is preserved.",
                    RuntimeWarning, stacklevel=2)
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
            # Keep temporary files on the target filesystem for atomic replace.
            with TemporaryDirectory(prefix=f".{subID}-", dir=filename.parent) as directory:
                temporary = Path(directory) / filename.name
                pot.export(str(temporary))
                Potential(str(temporary))  # Reject incomplete/unreadable exports.
                temporary.replace(filename)
        pot = Potential(str(filename))

    snapshot = Snapshot(basePath, snapNum)
    load_particle_fields = {"star": ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'GFM_StellarFormationTime'],
                            "dm": ['Coordinates', 'Velocities'],
                            "gas": ['Coordinates', 'Velocities', 'Masses']}
    snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=subID)
    snapshot.GC_physical_units()
    snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
    snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
    galaxy = snapshot.container
    X, model, galaxy, eoemin_cut, jzojc_cut, _ = decompose_prepared_galaxy(
        galaxy, potential=pot)
    if mixture_model_output_path is not None:
        mixture_model_output = util.decompose_mixture_model(model, eoemin_cut, jzojc_cut, -jzojc_cut)
        with open(f"{mixture_model_output_path}/mixture_model_{run}_{snapNum}_{subID}.pkl", "wb") as f:
            pickle.dump(mixture_model_output, f)
    if image_path is not None:
        import matplotlib.pyplot as plt
        from .visualization.decomposition import visualize_decomposition

        visualize_decomposition(X, model, galaxy, eoemin_cut, jzojc_cut, threshold_line=True, ranges=None)
        plt.savefig(Path(image_path)/f"{subID}.pdf", dpi=300, bbox_inches='tight')
    if structure_properties_output_path is not None:
        structure_properties_output = util.save_structure_properties(galaxy)
        with open(f"{structure_properties_output_path}/structure_properties_{run}_{snapNum}_{subID}.pkl", "wb") as f:
            pickle.dump(structure_properties_output, f)
    return model, galaxy, eoemin_cut, jzojc_cut
