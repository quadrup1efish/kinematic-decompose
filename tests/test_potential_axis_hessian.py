"""Exercise Cartesian axis Hessians in the exact loaded native extension."""
import ctypes
import importlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

_potential = importlib.import_module("kinematic_decompose.potential._potential")


PROBE = r'''
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
    source.write_text(PROBE)
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(compiler + ["-std=c++11", "-O2"] + link
                   + ["-I", str(include), str(source), "-o", str(library)], check=True)
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.axis_hessian_error.restype = ctypes.c_char_p
    assert probe.axis_hessian_regression() == 32, probe.axis_hessian_error().decode()
