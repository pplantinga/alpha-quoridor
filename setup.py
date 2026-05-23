import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("game.rules_fast", ["src/game/rules_fast.pyx"])
]

setup(
    name="alpha-quoridor",
    ext_modules=cythonize(extensions, annotate=True),
    package_dir={"": "src"},
    include_dirs=[numpy.get_include()]
)
