conda install --file requirements-conda.txt
pip install -r requirements.txt

jupyter notebook

# Create a distribution
python setup.py sdist bdist_wheel

# install from tar.gz
pip install O:\programs\perfect-physics\dist\perfect-physics-0.1.8.tar.gz

# upload to pypi
python -m twine upload dist/perfect_physics-0.1.Z*

pip install perfect_physics
pip uninstall perfect_physics

