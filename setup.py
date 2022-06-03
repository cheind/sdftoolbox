from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent


def read_requirements(fname):
    with open(THISDIR / "requirements" / fname, "r") as f:
        return f.read().splitlines()


core_required = read_requirements("requirements.txt")
dev_required = read_requirements("dev-requirements.txt") + core_required

main_ns = {}
with open(THISDIR / "surfacenets" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="sdf-surfacenets",
    version=main_ns["__version__"],
    description=(
        "Vectorized Python implementation of SurfaceNets for isosurface"
        " extraction."
    ),
    author="Christoph Heindl",
    url="https://github.com/cheind/sdf-surfacenets",
    license="MIT",
    install_requires=core_required,
    packages=find_packages(".", include="surfacenets*"),
    include_package_data=True,
    keywords="isoextraction isolevel surfacenets",
    extras_require={
        "dev": dev_required,
    },
)
