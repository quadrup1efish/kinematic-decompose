import hashlib
import os
import platform
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pybind11
from setuptools import Extension, setup

ROOT = Path(__file__).parent
NATIVE = ROOT / "src/kinematic_decompose/potential/native/agama/src"
SOURCES = [
    "coord.cpp", "math_core.cpp", "math_fit.cpp", "math_linalg.cpp", "math_specfunc.cpp",
    "math_sphharm.cpp", "math_spline.cpp", "potential_base.cpp",
    "potential_composite.cpp", "potential_multipole.cpp", "utils.cpp",
    "python_potential.cpp",
]
GSL_VERSION = "2.8"
GSL_SHA256 = "6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190"
GSL_URLS = (
    f"https://ftp.gnu.org/gnu/gsl/gsl-{GSL_VERSION}.tar.gz",
    f"https://ftpmirror.gnu.org/gnu/gsl/gsl-{GSL_VERSION}.tar.gz",
)
GSL_PACKAGE_DIR = ROOT / "src/kinematic_decompose/potential/native"
GSL_PACKAGE_ARCHIVE = GSL_PACKAGE_DIR / f"gsl-{GSL_VERSION}.tar.gz"
GSL_PACKAGE_COPYING = GSL_PACKAGE_DIR / "GSL-COPYING"


def _run_build_command(command, cwd, env, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        try:
            subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Bundled GSL build failed while running {command!r}. "
                f"See build log: {log_path}"
            ) from exc


def _download_gsl_archive(cache_root):
    archive = cache_root / f"gsl-{GSL_VERSION}.tar.gz"
    if archive.is_file() and hashlib.sha256(archive.read_bytes()).hexdigest() == GSL_SHA256:
        return archive
    archive.unlink(missing_ok=True)

    supplied_archive = os.environ.get("KINEMATIC_DECOMPOSE_GSL_ARCHIVE")
    candidates = ([Path(supplied_archive).expanduser()] if supplied_archive else [])
    if GSL_PACKAGE_ARCHIVE.is_file():
        candidates.append(GSL_PACKAGE_ARCHIVE)
    for candidate in candidates:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != GSL_SHA256:
            raise RuntimeError(
                f"GSL archive SHA-256 mismatch for {candidate}: "
                f"expected {GSL_SHA256}, got {digest}"
            )
        shutil.copyfile(candidate, archive)
        return archive

    errors = []
    for url in GSL_URLS:
        temporary = cache_root / f"gsl-{GSL_VERSION}.{os.getpid()}.download"
        try:
            urlretrieve(url, temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != GSL_SHA256:
                raise RuntimeError(f"SHA-256 mismatch for {url}: got {digest}")
            temporary.replace(archive)
            return archive
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not obtain verified GSL {GSL_VERSION} source. "
        "Check network access to GNU FTP/mirrors and retry, or set "
        "KINEMATIC_DECOMPOSE_GSL_ARCHIVE to a local copy of the official archive. Details: "
        + " | ".join(errors)
    )


def _build_static_gsl():
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Automatic bundled GSL builds currently support macOS and Linux.")

    cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_root = cache_base / "kinematic-decompose" / "gsl" / GSL_VERSION
    cache_root.mkdir(parents=True, exist_ok=True)
    build_flags = "\n".join((
        sysconfig.get_platform(), platform.machine(),
        os.environ.get("CC", sysconfig.get_config_var("CC") or "cc"),
        sysconfig.get_config_var("CFLAGS") or "", os.environ.get("CFLAGS", ""),
        os.environ.get("ARCHFLAGS", ""),
        os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "",
    ))
    build_key = hashlib.sha256(build_flags.encode()).hexdigest()[:16]
    install_prefix = cache_root / build_key
    gsl_include = install_prefix / "include"
    gsl_archive = install_prefix / "lib/libgsl.a"
    cblas_archive = install_prefix / "lib/libgslcblas.a"
    archive = _download_gsl_archive(cache_root)
    if (gsl_include / "gsl/gsl_errno.h").is_file() and gsl_archive.is_file() and cblas_archive.is_file():
        return gsl_include, gsl_archive, cblas_archive, archive

    work_dir = Path(tempfile.mkdtemp(prefix=f"gsl-{GSL_VERSION}-", dir=cache_root))
    log_path = cache_root / f"build-{build_key}.log"
    try:
        with tarfile.open(archive, "r:gz") as source_archive:
            source_archive.extractall(work_dir)
        source_dir = work_dir / f"gsl-{GSL_VERSION}"
        temporary_prefix = work_dir / "install"
        env = os.environ.copy()
        env["CC"] = os.environ.get("CC", sysconfig.get_config_var("CC") or "cc")
        env["CFLAGS"] = " ".join(filter(None, (
            sysconfig.get_config_var("CFLAGS") or "",
            os.environ.get("CFLAGS", ""),
            os.environ.get("ARCHFLAGS", ""),
            "-O2", "-fPIC",
        )))
        if platform.system() == "Darwin":
            deployment_target = (
                os.environ.get("MACOSX_DEPLOYMENT_TARGET")
                or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
            )
            if deployment_target:
                env["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
                env["CFLAGS"] += f" -mmacosx-version-min={deployment_target}"
        _run_build_command(
            ["./configure", f"--prefix={temporary_prefix}", "--disable-shared", "--enable-static"],
            source_dir, env, log_path,
        )
        _run_build_command(["make", "-j2"], source_dir, env, log_path)
        _run_build_command(["make", "install"], source_dir, env, log_path)
        if not (temporary_prefix / "include/gsl/gsl_errno.h").is_file():
            raise RuntimeError(f"GSL headers were not installed; see build log: {log_path}")
        if not (temporary_prefix / "lib/libgsl.a").is_file() or not (temporary_prefix / "lib/libgslcblas.a").is_file():
            raise RuntimeError(f"GSL static libraries were not installed; see build log: {log_path}")
        if install_prefix.exists():
            shutil.rmtree(install_prefix)
        temporary_prefix.replace(install_prefix)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return gsl_include, gsl_archive, cblas_archive, archive


def _stage_gsl_source_assets(source_archive):
    created = []
    try:
        if not GSL_PACKAGE_ARCHIVE.exists():
            shutil.copyfile(source_archive, GSL_PACKAGE_ARCHIVE)
            created.append(GSL_PACKAGE_ARCHIVE)
        elif hashlib.sha256(GSL_PACKAGE_ARCHIVE.read_bytes()).hexdigest() != GSL_SHA256:
            raise RuntimeError(f"Bundled GSL source has an unexpected checksum: {GSL_PACKAGE_ARCHIVE}")

        with tarfile.open(source_archive, "r:gz") as gsl_source:
            copying = next(
                member for member in gsl_source.getmembers()
                if member.isfile() and member.name.endswith("/COPYING")
            )
            copying_file = gsl_source.extractfile(copying)
            if copying_file is None:
                raise RuntimeError("GSL source archive does not contain a readable COPYING file")
            copying_contents = copying_file.read()
        if GSL_PACKAGE_COPYING.exists():
            if GSL_PACKAGE_COPYING.read_bytes() != copying_contents:
                raise RuntimeError(f"Bundled GSL license does not match GSL {GSL_VERSION}")
        else:
            GSL_PACKAGE_COPYING.write_bytes(copying_contents)
            created.append(GSL_PACKAGE_COPYING)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return created


GSL_INCLUDE, GSL_ARCHIVE, GSL_CBLAS_ARCHIVE, GSL_SOURCE_ARCHIVE = _build_static_gsl()
extension = Extension(
    "kinematic_decompose.potential._potential",
    [str(Path("src/kinematic_decompose/potential/native/agama/src") / source) for source in SOURCES],
    include_dirs=[str(NATIVE), pybind11.get_include(), str(GSL_INCLUDE)],
    extra_objects=[str(GSL_ARCHIVE), str(GSL_CBLAS_ARCHIVE)],
    libraries=["m"],
    language="c++",
    extra_compile_args=["-std=c++11", "-O2"],
)
temporary_source_assets = _stage_gsl_source_assets(GSL_SOURCE_ARCHIVE)
try:
    setup(
        ext_modules=[extension],
        include_package_data=True,
        package_data={
            "kinematic_decompose.potential": [
                "native/GSL-COPYING",
                f"native/gsl-{GSL_VERSION}.tar.gz",
            ],
        },
        zip_safe=False,
    )
finally:
    for path in temporary_source_assets:
        path.unlink(missing_ok=True)
