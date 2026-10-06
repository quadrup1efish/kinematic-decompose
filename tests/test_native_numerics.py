"""Native log-density, Cartesian Hessian and spline EDF regressions."""
import ctypes
import importlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import pytest


# Test density log recovery

"""Native regressions for the internal log-density recovery representation."""


_native = importlib.import_module('kinematic_decompose.potential._potential')
LOG_DENSITY_PROBE = r'''
#include "potential_multipole.h"
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
namespace { std::string error; }
extern "C" const char* recovery_error() { return error.c_str(); }
extern "C" int recovery_regression() {
 try {
  const math::SphHarmIndices ind(4,0,coord::ST_AXISYMMETRIC);
  std::vector<double> r(30);
  std::vector<std::vector<double> > logc(25,std::vector<double>(30,0.)), raw=logc;
  for(size_t k=0;k<r.size();k++) {
   double x=-5.+10.*k/(r.size()-1);r[k]=std::exp(x);
   logc[0][k]=-3*x-.5*std::pow((x-2.2)/.18,2);
   logc[6][k]=.1;logc[20][k]=.02;
   raw[0][k]=std::exp(logc[0][k]);raw[6][k]=.1*raw[0][k];raw[20][k]=.02*raw[0][k];
  }
  if(raw[0][0]!=0 || !(raw[0].back()>0)) throw std::runtime_error("fixture did not underflow only on the inner side");
  potential::DensitySphericalHarmonic legacy(r,raw), restored(r,logc,ind);
  if(legacy.hasFiniteMassTails() || !restored.hasFiniteMassTails()) throw std::runtime_error("divergent legacy tail was not distinguished from the log fit");
  double value[25];
  for(int j=0;j<121;j++) {
   restored.getCoefsAtRadius(j==0 ? 0. : std::exp(-9.+18.*j/120),value);
   for(int c=0;c<25;c++)if(!std::isfinite(value[c]))throw std::runtime_error("recovered evaluation is nonfinite");
   if(value[0]<0)throw std::runtime_error("negative recovered monopole");
  }
  for(size_t k=0;k<r.size();k++) {
   restored.getCoefsAtRadius(r[k],value);
   if(raw[0][k]>std::numeric_limits<double>::min()) {
    if(std::abs(value[0]/raw[0][k]-1)>1e-12 || std::abs(value[6]/value[0]-.1)>1e-12 || std::abs(value[20]/value[0]-.02)>1e-12)
     throw std::runtime_error("log-density or normalized harmonic changed at a fit node");
   }
  }
  // A true zero-density input remains an ordinary exact-zero density.
  std::vector<std::vector<double> > zeros(25,std::vector<double>(30,0.));
  potential::DensitySphericalHarmonic zero(r,zeros);
  for(double radius : {0.,.001,1.,10000.}) {
   zero.getCoefsAtRadius(radius,value);
   if(value[0]!=0)throw std::runtime_error("true zero density was made positive");
  }
  // Combine exponents before exp: an underflowed boundary can represent a
  // finite cusp density farther inward; exp(boundary)*pow() cannot recover it.
  for(size_t k=0;k<r.size();k++)logc[0][k]=-800.-std::log(r[k])-3*std::log1p(r[k]);
  potential::DensitySphericalHarmonic tiny(r,logc,ind);
  tiny.getCoefsAtRadius(1e-200,value);
  if(!(value[0]>0) || !std::isfinite(value[0]))throw std::runtime_error("premature exp underflow in cusp extrapolation");
  // Reject invalid original log fits, rather than clipping their slopes.
  for(int mode=0;mode<4;mode++) {
   std::vector<std::vector<double> > bad=logc;
   if(mode==0)for(size_t k=0;k<r.size();k++)bad[0][k]=0.;
   if(mode==1)for(size_t k=0;k<r.size();k++)bad[0][k]=-4*std::log(r[k]);
   if(mode==2)bad[0][0]=std::numeric_limits<double>::quiet_NaN();
   if(mode==3)bad[6][10]=std::numeric_limits<double>::infinity();
   bool rejected=false;
   try {potential::DensitySphericalHarmonic invalid(r,bad,ind);}
   catch(const std::domain_error&) {rejected=true;}
   if(!rejected)throw std::runtime_error("invalid log fit accepted or silently regularized");
  }
  return 0;
 } catch(const std::exception& e) {error=e.what();return -1;}
}
'''


def test_internal_log_density_recovery(tmp_path):
    module_file = _native.__file__
    assert module_file is not None
    binary = Path(module_file)
    include = binary.parent / 'native' / 'agama' / 'src'
    compiler = shlex.split(os.environ.get('CXX') or 'c++')
    if sys.platform not in ('darwin', 'linux') or not (include / 'potential_multipole.h').is_file() or not compiler or not shutil.which(compiler[0]):
        pytest.skip('native source headers and a POSIX C++ compiler are required')
    source = tmp_path / 'density-recovery.cpp'
    library = tmp_path / 'density-recovery.so'
    source.write_text(LOG_DENSITY_PROBE)
    link = ['-bundle', '-undefined', 'dynamic_lookup'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    subprocess.run(compiler + ['-std=c++11', '-O2'] + link + ['-I', str(include), str(source), '-o', str(library)], check=True)
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.recovery_error.restype = ctypes.c_char_p
    assert probe.recovery_regression() == 0, probe.recovery_error().decode()


# Test potential axis hessian

"""Exercise Cartesian axis Hessians in the exact loaded native extension."""


_potential = importlib.import_module("kinematic_decompose.potential._potential")


AXIS_HESSIAN_PROBE = r'''
#include "potential_multipole.h"
#include "potential_composite.h"
#include <cmath>
#include <string>
#include <stdexcept>
namespace { std::string error; }
extern "C" const char* axis_hessian_error() { return error.c_str(); }
extern "C" int axis_hessian_regression() {
 try {
  particles::ParticleArray<coord::PosCyl> source;
  source.add(coord::toPosCyl(coord::PosCar(.2,-.1,.4)),1.);
  source.add(coord::toPosCyl(coord::PosCar(1.2,.8,-.5)),2.);
  source.add(coord::toPosCyl(coord::PosCar(-.7,.3,1.1)),.7);
  source.add(coord::toPosCyl(coord::PosCar(.05,.2,.15)),.2);
  int checked=0;
  for(int softened=0; softened<2; softened++) {
   for(int triaxial=0; triaxial<2; triaxial++) {
    coord::SymmetryType sym=triaxial ? coord::ST_TRIAXIAL : coord::ST_NONE;
    potential::PtrPotential single=softened ?
     potential::Multipole::createSoftened(source,std::vector<double>(1,.4),sym,4,4,60,.02,20) :
     potential::Multipole::create(source,sym,4,4,60,.02,20);
    std::vector<potential::PtrPotential> parts(2,single);
    potential::PtrPotential composite(new potential::Composite(parts));
    for(int combined=0; combined<2; combined++) {
     const potential::BasePotential& pot=*(combined ? composite : single);
     for(double z : {1.,-1.,40.,-40.}) {
      coord::PosCar p(0,0,z);
      coord::HessCar hessian, hessianOnly;
      coord::GradCar gradient;
      pot.eval(p,NULL,&gradient,&hessian);
      // Hessian-only calls must work without an output gradient buffer.
      pot.eval(p,NULL,NULL,&hessianOnly);
      double actual[3][3]={{hessian.dx2,hessian.dxdy,hessian.dxdz},
                          {hessian.dxdy,hessian.dy2,hessian.dydz},
                          {hessian.dxdz,hessian.dydz,hessian.dz2}};
      double only[6]={hessianOnly.dx2,hessianOnly.dy2,hessianOnly.dz2,
                      hessianOnly.dxdy,hessianOnly.dxdz,hessianOnly.dydz};
      double full[6]={hessian.dx2,hessian.dy2,hessian.dz2,
                      hessian.dxdy,hessian.dxdz,hessian.dydz};
      for(int j=0;j<6;j++)
       if(!std::isfinite(full[j]) || full[j]!=only[j])
        throw std::runtime_error("Hessian-only dispatch mismatch");
      double scale=0;
      for(int i=0;i<3;i++)for(int j=0;j<3;j++)
       scale=std::max(scale,std::abs(actual[i][j]));
      // Use two decreasing steps, testing every matrix entry rather than
      // only its trace. Compare the derivative of the independently
      // evaluated Cartesian gradient, not a density-derived Hessian.
      for(double relativeStep : {1e-5,1e-6}) {
       double step=std::abs(z)*relativeStep;
       for(int axis=0;axis<3;axis++) {
        coord::PosCar plus=p,minus=p;
        if(axis==0){plus.x+=step;minus.x-=step;}
        if(axis==1){plus.y+=step;minus.y-=step;}
        if(axis==2){plus.z+=step;minus.z-=step;}
        coord::GradCar gp,gm;
        pot.eval(plus,NULL,&gp);pot.eval(minus,NULL,&gm);
        double fd[3]={(gp.dx-gm.dx)/(2*step),(gp.dy-gm.dy)/(2*step),
                      (gp.dz-gm.dz)/(2*step)};
        for(int component=0;component<3;component++) {
         double tolerance=1e-4*scale+1e-10;
         if(std::abs(actual[component][axis]-fd[component])>tolerance)
          throw std::runtime_error("Axis Hessian disagrees with gradient derivative: z="+
           std::to_string(z)+" row="+std::to_string(component)+" col="+std::to_string(axis));
        }
       }
      }
      double trace=hessian.dx2+hessian.dy2+hessian.dz2;
      if(std::abs(trace/(4*M_PI)-pot.density(p))>1e-8*scale+1e-12)
       throw std::runtime_error("Axis Hessian trace disagrees with density");
      checked++;
     }
    }
   }
  }
  return checked;
 } catch(const std::exception& e) {error=e.what();return -1;}
}
'''


def test_native_cartesian_axis_hessian_all_components(tmp_path):
    module_file = _potential.__file__
    assert module_file is not None
    binary = Path(module_file)
    include = binary.parent / "native" / "agama" / "src"
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    if (sys.platform not in ("darwin", "linux") or not compiler
            or not shutil.which(compiler[0]) or not (include / "potential_multipole.h").is_file()):
        pytest.skip("native headers and a POSIX C++ compiler are required")
    source = tmp_path / "axis-hessian.cpp"
    library = tmp_path / "axis-hessian.so"
    source.write_text(AXIS_HESSIAN_PROBE)
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(compiler + ["-std=c++11", "-O2"] + link
                   + ["-I", str(include), str(source), "-o", str(library)], check=True)
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.axis_hessian_error.restype = ctypes.c_char_p
    assert probe.axis_hessian_regression() == 32, probe.axis_hessian_error().decode()


# Test spline prepared edf

"""Verify fixed-EDF preparation against the ordinary native spline fit."""


_potential = importlib.import_module("kinematic_decompose.potential._potential")


SPLINE_EDF_PROBE = r'''
#include "math_spline.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

namespace { std::string error; }
extern "C" const char* probe_error() { return error.c_str(); }
extern "C" int prepared_edf_size() { return sizeof(math::SplineApprox::PreparedEDF); }
extern "C" int run_regression() {
    try {
        int checked = 0;
        for(int knots : {2, 3, 4, 8, 30}) {
            std::vector<double> grid(knots), x(1500), y(1500);
            grid[0] = -2.;
            for(int i=1; i<knots; i++) grid[i] = grid[i-1] + .1 + std::exp(.8*std::sin(i));
            for(size_t i=0; i<x.size(); i++) {
                x[i] = grid.front() + (grid.back()-grid.front()) *
                    (1.2*std::fmod(i*.6180339887498949, 1.) - .1);
                y[i] = std::sin(x[i]) + .03*std::cos(i);
            }
            for(int mode=0; mode<3; mode++) {
                std::vector<double> weights;
                if(mode) {
                    weights.resize(x.size());
                    for(size_t i=0; i<weights.size(); i++)
                        weights[i] = mode==2 && i%7==0 ? 0. : .5+std::fmod(i*.3127,1.);
                }
                math::SplineApprox fitter(grid, x, weights);
                // Fitting must not depend on the lifetime or subsequent content of source knots/coordinates.
                std::unique_ptr<math::SplineApprox> detached;
                {
                    std::vector<double> sourceGrid = grid;
                    std::vector<double> source = x;
                    detached.reset(new math::SplineApprox(sourceGrid, source, weights));
                    std::fill(sourceGrid.begin(), sourceGrid.end(), NAN);
                    std::fill(source.begin(), source.end(), NAN);
                }
                double referenceRms, detachedRms;
                const std::vector<double> referenceValues = fitter.fit(y, 2.+.5*(knots-2), &referenceRms);
                const std::vector<double> detachedValues = detached->fit(y, 2.+.5*(knots-2), &detachedRms);
                if(referenceValues.size()!=detachedValues.size() ||
                    std::memcmp(referenceValues.data(), detachedValues.data(), referenceValues.size()*sizeof(double)) ||
                    std::memcmp(&referenceRms, &detachedRms, sizeof(double)))
                    throw std::runtime_error("constructor-input lifetime dependency");
                checked++;
                for(double edf : {0., 2., 2.+.25*(knots-2), 2.+.75*(knots-2),
                    static_cast<double>(knots), -1., knots+1., INFINITY*1., NAN*1.}) {
                    std::vector<double> a, b;
                    double ar=0., br=0.;
                    std::string ae, be;
                    try { a = fitter.fit(y, edf, &ar); }
                    catch(const std::exception& e) { ae = std::string(typeid(e).name())+":"+e.what(); }
                    try { b = fitter.fit(y, fitter.prepareEDF(edf), &br); }
                    catch(const std::exception& e) { be = std::string(typeid(e).name())+":"+e.what(); }
                    if(ae!=be || a.size()!=b.size() ||
                        (!a.empty() && std::memcmp(a.data(), b.data(), a.size()*sizeof(double))) ||
                        (ae.empty() && std::memcmp(&ar, &br, sizeof(double))))
                        throw std::runtime_error("ordinary/prepared fit mismatch");
                    checked++;
                }
                for(int defect=0; defect<2; defect++) {
                    std::vector<double> bad = y;
                    if(defect==0) bad.pop_back();
                    else bad[19] = NAN;
                    std::string ae, be;
                    std::vector<double> a, b;
                    try { a = fitter.fit(bad, 2.+.5*(knots-2)); }
                    catch(const std::exception& e) { ae = std::string(typeid(e).name())+":"+e.what(); }
                    try { b = fitter.fit(bad, fitter.prepareEDF(2.+.5*(knots-2))); }
                    catch(const std::exception& e) { be = std::string(typeid(e).name())+":"+e.what(); }
                    if(ae!=be || a.size()!=b.size() ||
                        (!a.empty() && std::memcmp(a.data(), b.data(), a.size()*sizeof(double))))
                        throw std::runtime_error("data-validation mismatch");
                    checked++;
                }
                const math::SplineApprox::PreparedEDF prepared = fitter.prepareEDF(2.+.5*(knots-2));
                math::SplineApprox another(grid, x, weights);
                bool rejected = false;
                try { another.fit(y, prepared); }
                catch(const std::invalid_argument&) { rejected = true; }
                if(!rejected) throw std::runtime_error("cross-fitter preparation accepted");
                checked++;
                const std::vector<double> reference = fitter.fit(y, prepared);
                std::vector<std::vector<double> > result(4);
                std::vector<std::thread> workers;
                for(size_t i=0; i<result.size(); i++)
                    workers.push_back(std::thread([&, i]() { result[i] = fitter.fit(y, prepared); }));
                for(size_t i=0; i<workers.size(); i++) workers[i].join();
                for(size_t i=0; i<result.size(); i++) {
                    if(result[i].size()!=reference.size() ||
                        std::memcmp(result[i].data(), reference.data(), reference.size()*sizeof(double)))
                        throw std::runtime_error("shared preparation concurrency mismatch");
                    checked++;
                }
            }
        }
        return checked;
    }
    catch(const std::exception& e) { error=e.what(); return -1; }
}
'''


def test_prepared_edf_matches_ordinary_fit(tmp_path):
    module_file = _potential.__file__
    assert module_file is not None
    include = Path(module_file).parent / "native" / "agama" / "src"
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("native probe requires a POSIX shared-library compiler")
    if not compiler or not (include / "math_spline.h").is_file() or not shutil.which(compiler[0]):
        pytest.skip("native source headers and a C++ compiler are required")
    source = tmp_path / "prepared-edf.cpp"
    library = tmp_path / "prepared-edf.so"
    source.write_text(SPLINE_EDF_PROBE)
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared", "-fPIC", "-pthread"]
    subprocess.run(compiler + ["-std=c++11", "-O2"] + link +
                   ["-I", str(include), str(source), "-o", str(library)], check=True)
    # The probe resolves native C++ symbols from this exact extension.
    ctypes.CDLL(module_file, mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.probe_error.restype = ctypes.c_char_p
    assert probe.run_regression() == 5 * 3 * (9 + 2 + 1 + 4 + 1), probe.probe_error().decode()
    assert probe.prepared_edf_size() <= 2 * max(ctypes.sizeof(ctypes.c_void_p), ctypes.sizeof(ctypes.c_double))
