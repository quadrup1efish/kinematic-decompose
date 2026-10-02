"""Verify fixed-EDF preparation against the ordinary native spline fit."""
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
    source.write_text(PROBE)
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared", "-fPIC", "-pthread"]
    subprocess.run(compiler + ["-std=c++11", "-O2"] + link +
                   ["-I", str(include), str(source), "-o", str(library)], check=True)
    # The probe resolves native C++ symbols from this exact extension.
    ctypes.CDLL(module_file, mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(library))
    probe.probe_error.restype = ctypes.c_char_p
    assert probe.run_regression() == 5 * 3 * (9 + 2 + 1 + 4 + 1), probe.probe_error().decode()
    assert probe.prepared_edf_size() <= 2 * max(ctypes.sizeof(ctypes.c_void_p), ctypes.sizeof(ctypes.c_double))
