# Spatial-region surface densities

Reproduce: `.venv/bin/python tests/example_load_region.py`

Source: `/Users/yuwa/sims.TNG/TNG100-3/output`, snapshot 99; a=0.9999999999999998, h=0.6774.
Center (ckpc/h): [850.66307177, 26334.70465888, 18281.44870506].
Physical field: 20.0 x 20.0 Mpc; depth: 10.0 Mpc.
Resolution: 512 squared; pixel: 39.0625 kpc.
Projection: box z axis, periodic spatial selection, no halo membership or rotation.
Renderer: pynbody.plot.sph.image, rho, Msol kpc^-2, unweighted integral.
Smoothing: DM/gas adaptive 32 neighbours; stars fixed to one pixel.

| Family | N | Particle mass [Msol] | Image integral [Msol] | Difference [%] | Color limits [Msol/kpc^2] |
|---|---:|---:|---:|---:|---|
| dm | 2215700 | 1.05835338907e+15 | 1.05622250228e+15 | -0.201340 | [40136.257421875, 225762423.4559946] |
| gas | 2020244 | 1.90134878938e+14 | 1.89952489123e+14 | -0.095927 | [7456.617763671875, 30140568.759999838] |
| star | 92372 | 6.04383185951e+12 | 6.08838128276e+12 | +0.737106 | [0.03742083609104156, 30684482.991998218] |
