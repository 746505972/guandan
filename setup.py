# 2025/4/2 18:10
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["c_give_cards.pyx"], language_level="3"))
setup(ext_modules=cythonize("c_rule.pyx", language_level="3"))