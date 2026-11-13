"""Build optional Fortran/OpenMP extension in work/, copy only final module."""
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
build=root/'work/python-native';build.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','numpy.f2py','-c',str(root/'python/native/fourier.f90'),'-m','_tbkit_fortran',
                '--backend','meson','--f90flags=-O2 -fopenmp','-lgomp'],cwd=build,check=True)
modules=list(build.glob('_tbkit_fortran*.so'))
if len(modules)!=1:raise RuntimeError('expected one compiled extension module')
destination=root/'python/tbkit'/modules[0].name
shutil.copy2(modules[0],destination)
print(destination)
