#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "potential_composite.h"
#include "potential_multipole.h"
#include "units.h"

#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using potential::PtrPotential;

namespace {

struct PotentialHandle {
    PtrPotential potential;
};

bool isSupportedPotential(const potential::BasePotential& potential)
{
    if(dynamic_cast<const potential::Multipole*>(&potential))
        return true;
    const potential::BaseComposite<potential::BasePotential>* composite =
        dynamic_cast<const potential::BaseComposite<potential::BasePotential>*>(&potential);
    if(!composite)
        return false;
    for(unsigned int i=0; i<composite->size(); i++)
        if(!isSupportedPotential(*composite->component(i)))
            return false;
    return true;
}

const units::InternalUnits internalUnits(2.7183 * units::Kpc, 3.1416 * units::Myr);
double lengthUnit = 1., velocityUnit = 1., massUnit = 1., timeUnit = 1.;
bool potentialCreated = false;

units::ExternalUnits externalUnits()
{
    if(lengthUnit==1. && velocityUnit==1. && massUnit==1. && timeUnit==1.)
        return units::ExternalUnits();
    return units::ExternalUnits(internalUnits,
        lengthUnit * units::Kpc / internalUnits.from_Kpc,
        velocityUnit * units::kms / internalUnits.from_kms,
        massUnit * units::Msun / internalUnits.from_Msun);
}

double gravitationalConstant()
{
    if(lengthUnit==1. && velocityUnit==1. && massUnit==1. && timeUnit==1.)
        return 1.;
    const units::ExternalUnits conv = externalUnits();
    return units::Grav * (conv.massUnit * internalUnits.to_Msun * units::Msun) /
        (conv.velocityUnit * internalUnits.to_kms * units::kms) /
        (conv.velocityUnit * internalUnits.to_kms * units::kms) /
        (conv.lengthUnit * internalUnits.to_Kpc * units::Kpc);
}

double setUnits(double mass, double length, double velocity, double time)
{
    const bool reset = mass==0 && length==0 && velocity==0 && time==0;
    if(mass<0 || length<0 || velocity<0 || time<0)
        throw std::invalid_argument("Invalid arguments passed to setUnits()");
    if(length>0 && velocity>0 && time>0)
        throw std::invalid_argument("You may not assign length, velocity and time units simultaneously");
    if(mass==0 && !reset)
        throw std::invalid_argument("You must specify mass unit");
    if(reset) {
        lengthUnit=velocityUnit=massUnit=timeUnit=1.;
    } else if(length>0 && time>0) {
        const units::ExternalUnits conv(internalUnits,
            length*units::Kpc, length/time*units::Kpc/units::Myr, mass*units::Msun);
        lengthUnit=conv.lengthUnit; velocityUnit=conv.velocityUnit;
        massUnit=conv.massUnit; timeUnit=conv.timeUnit;
    } else if(length>0 && velocity>0) {
        const units::ExternalUnits conv(internalUnits,
            length*units::Kpc, velocity*units::kms, mass*units::Msun);
        lengthUnit=conv.lengthUnit; velocityUnit=conv.velocityUnit;
        massUnit=conv.massUnit; timeUnit=conv.timeUnit;
    } else if(time>0 && velocity>0) {
        const units::ExternalUnits conv(internalUnits,
            velocity*time*units::kms*units::Myr, velocity*units::kms, mass*units::Msun);
        lengthUnit=conv.lengthUnit; velocityUnit=conv.velocityUnit;
        massUnit=conv.massUnit; timeUnit=conv.timeUnit;
    } else {
        throw std::invalid_argument("You must specify exactly two out of three units: length, time and velocity");
    }
    return gravitationalConstant();
}

coord::SymmetryType parseSymmetry(const std::string& name)
{
    if(name.empty()) return coord::ST_TRIAXIAL;
    switch(std::tolower(name[0])) {
        case 's': return coord::ST_SPHERICAL;
        case 'a': return coord::ST_AXISYMMETRIC;
        case 't': return coord::ST_TRIAXIAL;
        case 'n': return coord::ST_NONE;
        default: throw std::invalid_argument("Unsupported Multipole symmetry: " + name);
    }
}

std::string symmetryName(coord::SymmetryType symmetry)
{
    if(symmetry==coord::ST_SPHERICAL) return "Spherical";
    if(symmetry==coord::ST_AXISYMMETRIC) return "Axisymmetric";
    if(symmetry==coord::ST_TRIAXIAL) return "Triaxial";
    if(symmetry==coord::ST_NONE) return "None";
    return std::to_string(static_cast<int>(symmetry));
}

std::string trim(const std::string& text)
{
    const size_t first=text.find_first_not_of(" \t\r\n");
    if(first==std::string::npos) return "";
    const size_t last=text.find_last_not_of(" \t\r\n");
    return text.substr(first, last-first+1);
}

using Coefficients = std::vector<std::vector<double> >;

void writeHarmonics(std::ostream& out, const std::vector<double>& radii,
    const Coefficients& coefficients)
{
    const int lmax=static_cast<int>(std::sqrt(coefficients.size())-1);
    out << "#radius";
    for(int l=0; l<=lmax; ++l)
        for(int m=-l; m<=l; ++m)
            out << "\tl=" << l << ",m=" << m;
    out << '\n' << std::setprecision(15);
    for(size_t r=0; r<radii.size(); ++r) {
        out << radii[r];
        for(const auto& coefficient: coefficients)
            out << '\t' << (r<coefficient.size() ? coefficient[r] : 0.);
        out << '\n';
    }
}

void writeMultipole(std::ostream& out, const potential::Multipole& multipole,
    const units::ExternalUnits& converter)
{
    std::vector<double> radii;
    Coefficients phi, dphi;
    multipole.getCoefs(radii, phi, dphi);
    for(double& r: radii) r /= converter.lengthUnit;
    for(Coefficients* array: {&phi, &dphi})
        for(auto& row: *array)
            for(double& value: row)
                value /= converter.velocityUnit * converter.velocityUnit /
                    (array==&dphi ? converter.lengthUnit : 1.);
    out << "gridSizeR=" << radii.size() << "\nlmax="
        << static_cast<int>(std::sqrt(phi.size())-1) << "\nsymmetry="
        << symmetryName(multipole.symmetry()) << "\nCoefficients\n#Phi\n";
    writeHarmonics(out, radii, phi);
    out << "\n#dPhi/dr\n";
    writeHarmonics(out, radii, dphi);
}

void writePotential(std::ostream& out, const potential::BasePotential& pot,
    const units::ExternalUnits& converter, unsigned int& section)
{
    const auto* composite=dynamic_cast<const potential::BaseComposite<potential::BasePotential>*>(&pot);
    if(composite) {
        for(unsigned int i=0; i<composite->size(); ++i)
            writePotential(out, *composite->component(i), converter, section);
        return;
    }
    const auto* multipole=dynamic_cast<const potential::Multipole*>(&pot);
    if(!multipole)
        throw std::invalid_argument("Only Multipole and Composite potentials can be exported");
    out << "[Potential";
    if(section) out << section;
    ++section;
    out << "]\ntype=Multipole\n";
    writeMultipole(out, *multipole, converter);
    out << '\n';
}

bool writePotentialFile(const std::string& path, const potential::BasePotential& pot)
{
    std::ofstream out(path.c_str());
    if(!out) return false;
    unsigned int section=0;
    writePotential(out, pot, externalUnits(), section);
    return out.good();
}

Coefficients readHarmonics(const std::vector<std::string>& lines, size_t& index,
    size_t gridSize, size_t coefficientCount, std::vector<double>* radii)
{
    while(index<lines.size() && lines[index].find("#radius")!=0) ++index;
    if(index==lines.size()) throw std::runtime_error("Missing #radius coefficient header");
    ++index;
    Coefficients result(coefficientCount);
    for(size_t row=0; row<gridSize; ++row) {
        while(index<lines.size() && trim(lines[index]).empty()) ++index;
        if(index>=lines.size() || lines[index][0]=='#')
            throw std::runtime_error("Incomplete Multipole coefficient table");
        std::istringstream stream(lines[index++]);
        double radius;
        if(!(stream >> radius)) throw std::runtime_error("Invalid Multipole radius row");
        if(radii) radii->push_back(radius*lengthUnit);
        for(size_t c=0; c<coefficientCount; ++c) {
            double value;
            if(!(stream >> value)) throw std::runtime_error("Invalid Multipole coefficient row");
            result[c].push_back(value);
        }
    }
    return result;
}

PtrPotential readPotential(const std::string& path, const units::ExternalUnits& converter)
{
    std::ifstream input(path.c_str());
    if(!input) throw std::runtime_error("Cannot open potential file: " + path);
    std::vector<std::string> lines;
    for(std::string line; std::getline(input, line);) lines.push_back(line);
    std::vector<PtrPotential> components;
    for(size_t start=0; start<lines.size(); ++start) {
        if(lines[start].find("[Potential")!=0) continue;
        std::map<std::string, std::string> params;
        size_t coeff=start+1;
        for(; coeff<lines.size() && lines[coeff].find('[')!=0; ++coeff) {
            const size_t equal=lines[coeff].find('=');
            if(equal!=std::string::npos)
                params[trim(lines[coeff].substr(0,equal))]=trim(lines[coeff].substr(equal+1));
            if(trim(lines[coeff])=="Coefficients") { ++coeff; break; }
        }
        if(params["type"]!="Multipole")
            throw std::runtime_error("Only Agama Multipole and Composite INI files are supported");
        const size_t nr=static_cast<size_t>(std::stoul(params.at("gridSizeR")));
        const size_t lmax=static_cast<size_t>(std::stoul(params.at("lmax")));
        const size_t nc=(lmax+1)*(lmax+1);
        std::vector<double> radii;
        while(coeff<lines.size() && trim(lines[coeff]).empty()) ++coeff;
        if(coeff>=lines.size() || trim(lines[coeff++])!="#Phi")
            throw std::runtime_error("Missing #Phi coefficient block");
        Coefficients phi=readHarmonics(lines, coeff, nr, nc, &radii);
        while(coeff<lines.size() && trim(lines[coeff]).empty()) ++coeff;
        if(coeff>=lines.size() || trim(lines[coeff++])!="#dPhi/dr")
            throw std::runtime_error("Missing #dPhi/dr coefficient block");
        Coefficients dphi=readHarmonics(lines, coeff, nr, nc, nullptr);
        for(auto* array: {&phi, &dphi})
            for(auto& row: *array)
                for(double& value: row)
                    value *= converter.velocityUnit*converter.velocityUnit /
                        (array==&dphi ? converter.lengthUnit : 1.);
        components.push_back(PtrPotential(new potential::Multipole(radii, phi, dphi)));
        start=coeff;
    }
    if(components.empty()) throw std::runtime_error("No [Potential] Multipole sections found");
    if(components.size()==1) return components[0];
    return PtrPotential(new potential::Composite(components));
}

PtrPotential makeMultipole(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses,
    py::object softening, const std::string& symmetry, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    auto p = positions.request(), m = masses.request();
    if(p.ndim != 2 || p.shape[1] != 3 || m.ndim != 1 || p.shape[0] != m.shape[0])
        throw std::invalid_argument("positions must be (N,3) and masses must be (N,)");
    const size_t n = static_cast<size_t>(p.shape[0]);
    const double* pp = static_cast<const double*>(p.ptr);
    const double* mm = static_cast<const double*>(m.ptr);
    particles::ParticleArray<coord::PosCyl> particles;
    for(size_t i=0; i<n; ++i) {
        const double x=pp[3*i], y=pp[3*i+1], z=pp[3*i+2];
        particles.add(coord::PosCyl(std::hypot(x,y)*lengthUnit, z*lengthUnit,
            std::atan2(y,x)), mm[i]*massUnit);
    }
    std::vector<double> eps;
    if(!softening.is_none()) {
        if(py::isinstance<py::array>(softening))
            eps = softening.cast<std::vector<double> >();
        else
            eps.push_back(softening.cast<double>());
        if(eps.size()!=1 && eps.size()!=n)
            throw std::invalid_argument("softening must be scalar or have one value per particle");
        for(double& value: eps) {
            if(!(value>0) || !std::isfinite(value))
                throw std::invalid_argument("softening values must be finite and positive");
            value *= lengthUnit;
        }
        return potential::Multipole::createSoftened(particles, eps, parseSymmetry(symmetry),
            lmax, mmax, gridSizeR, rmin*lengthUnit, rmax*lengthUnit);
    }
    return potential::Multipole::create(particles, parseSymmetry(symmetry), lmax, mmax,
        gridSizeR, rmin*lengthUnit, rmax*lengthUnit);
}

py::array_t<double> evaluate(const PtrPotential& pot,
    py::array_t<double, py::array::c_style | py::array::forcecast> points, bool wantForce)
{
    auto p=points.request();
    if(p.ndim!=2 || p.shape[1]!=3)
        throw std::invalid_argument("coordinates must have shape (N,3)");
    const size_t n=static_cast<size_t>(p.shape[0]);
    const double* xyz=static_cast<const double*>(p.ptr);
    if(wantForce) {
        py::array_t<double> out({n, size_t(3)});
        double* result=out.mutable_data();
        for(size_t i=0; i<n; ++i) {
            coord::GradCar grad;
            pot->eval(coord::PosCar(xyz[3*i]*lengthUnit,xyz[3*i+1]*lengthUnit,
                xyz[3*i+2]*lengthUnit), nullptr, &grad);
            const double scale=lengthUnit/(velocityUnit*velocityUnit);
            result[3*i]=-grad.dx*scale; result[3*i+1]=-grad.dy*scale; result[3*i+2]=-grad.dz*scale;
        }
        return out;
    }
    py::array_t<double> out(n);
    double* result=out.mutable_data();
    for(size_t i=0; i<n; ++i)
        result[i]=pot->value(coord::PosCar(xyz[3*i]*lengthUnit,xyz[3*i+1]*lengthUnit,
            xyz[3*i+2]*lengthUnit))/(velocityUnit*velocityUnit);
    return out;
}

} // namespace

PYBIND11_MODULE(_potential, m)
{
    m.doc() = "Narrow Agama-compatible native Multipole and Composite backend";
    m.def("setUnits", [](double mass, double length, double velocity, double time) {
        const double oldLength=lengthUnit, oldVelocity=velocityUnit;
        const double oldMass=massUnit, oldTime=timeUnit;
        const double result=setUnits(mass, length, velocity, time);
        if(potentialCreated && (oldLength!=lengthUnit || oldVelocity!=velocityUnit ||
            oldMass!=massUnit || oldTime!=timeUnit) &&
            PyErr_WarnEx(PyExc_RuntimeWarning,
                "setUnits() called after creating Potential instances may cause incorrect I/O scaling", 1)<0)
            throw py::error_already_set();
        return result;
    }, py::arg("mass")=0., py::arg("length")=0., py::arg("velocity")=0., py::arg("time")=0.);
    m.def("getUnits", []() {
        py::dict result;
        if(lengthUnit==1. && velocityUnit==1. && timeUnit==1. && massUnit==1.)
            return result;
        result["length"] = lengthUnit * internalUnits.to_Kpc;
        result["velocity"] = velocityUnit * internalUnits.to_kms;
        result["time"] = timeUnit * internalUnits.to_Myr;
        result["mass"] = massUnit * internalUnits.to_Msun;
        return result;
    });
    m.def("gravitational_constant", &gravitationalConstant);
    py::class_<PotentialHandle>(m, "_BasePotential")
        .def("potential", [](const PotentialHandle& self, py::array_t<double, py::array::c_style | py::array::forcecast> xyz) {
            return evaluate(self.potential, xyz, false);
        })
        .def("force", [](const PotentialHandle& self, py::array_t<double, py::array::c_style | py::array::forcecast> xyz) {
            return evaluate(self.potential, xyz, true);
        })
        .def("export", [](const PotentialHandle& self, const std::string& path) {
            if(!writePotentialFile(path, *self.potential))
                throw std::runtime_error("failed to export potential: " + path);
        });
    m.def("multipole", [](py::array_t<double, py::array::c_style | py::array::forcecast> positions,
        py::array_t<double, py::array::c_style | py::array::forcecast> masses,
        py::object softening, const std::string& symmetry, int lmax, int mmax,
        unsigned int gridSizeR, double rmin, double rmax) {
        PtrPotential result=makeMultipole(positions, masses, softening, symmetry,
            lmax, mmax, gridSizeR, rmin, rmax);
        potentialCreated=true;
        return PotentialHandle{result};
    }, py::arg("positions"), py::arg("masses"),
        py::arg("softening")=py::none(), py::arg("symmetry")="a", py::arg("lmax")=4,
        py::arg("mmax")=4, py::arg("gridSizeR")=40, py::arg("rmin")=0., py::arg("rmax")=0.);
    m.def("composite", [](const std::vector<PotentialHandle>& handles) {
        std::vector<PtrPotential> parts;
        for(const PotentialHandle& handle: handles)
            parts.push_back(handle.potential);
        potentialCreated=true;
        return PotentialHandle{std::make_shared<potential::Composite>(parts)};
    });
    m.def("load", [](const std::string& path) {
        PtrPotential result=readPotential(path, externalUnits());
        if(!isSupportedPotential(*result))
            throw std::invalid_argument("Only Agama Multipole and Composite potential files are supported");
        potentialCreated=true;
        return PotentialHandle{result};
    });
}
