"""Native regressions for the internal log-density recovery representation."""
import ctypes
import importlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

_native = importlib.import_module('kinematic_decompose.potential._potential')
PROBE = r'''
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
    source.write_text(PROBE)
    link = ['-bundle', '-undefined', 'dynamic_lookup'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    subprocess.run(compiler + ['-std=c++11', '-O2'] + link + ['-I', str(include), str(source), '-o', str(library)], check=True)
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.recovery_error.restype = ctypes.c_char_p
    assert probe.recovery_regression() == 0, probe.recovery_error().decode()
