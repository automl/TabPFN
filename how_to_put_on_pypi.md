This file is only relevant for contributors, that upload to PyPI.
The `pyproject.toml` is the important setting for Pypi, the setup.py is only important for local installations.
When you want to upload a new version:
0. Have an environment just for these uploads (`TabPFNPipPackage`), activate it with `conda activate TabPFNPipPackage`.
1. run all tests in the `tests` folder.
2. update the version in `pyproject.toml`.
3. delete previous builds `rm dist/*`
4. `python3 -m pip install --upgrade build; python3 -m build`
5. test upload `python3 -m twine upload --repository testpypi dist/*`
6. download with `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tabpfn`, go back if something is broken
7. `python3 -m pip install --upgrade twine; python3 -m twine upload --repository pypi dist/*`

This should be it. Try the new version, ideally in a super fresh setup, like in a new colab.
