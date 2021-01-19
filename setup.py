from setuptools import setup

setup(name='gafs',
      version='0.0.1.1',
      description='Genetic algorigthm feature selection.',
      url='https://github.com/Shemka/GAFS',
      packages=['gafs'],
      author='Alexander Shemchuk',
      author_email='sheminy32@gmail.com',
      license='MIT',
      install_requires=['tqdm>=4.56.0',
                        'numpy>=1.19.5',
                        'scikit-learn>=0.24.0'])