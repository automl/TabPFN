from setuptools import setup

setup(name='tabpfn',
      version='0.1',
      description='Interface for using TabPFN and library to train TabPFN',
      url='https://github.com/automl/TabPFN',
      author='Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter',
      author_email='muellesa@tf.uni-freiburg.de',
      license='MIT',
      packages=['tabpfn'],
      python_requires='>=3.7',
      install_requires=[
        'gpytorch>=1.5.0',
        'torch>=1.9.0',
        'scikit-learn>=0.24.2',
        'pyyaml>=5.4.1',
        'seaborn>=0.11.2',
        'xgboost>=1.4.0',
        'tqdm>=4.62.1',
        'numpy>=1.21.2',
        'openml>=0.12.2',
        'catboost>=0.26.1',
        'auto-sklearn>=0.14.5',
        'hyperopt>=0.2.5',
        'configspace>=0.4.21',
      ],
      zip_safe=False)

