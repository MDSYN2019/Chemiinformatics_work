from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = 'chem2quant',
      version = '0.1',
      description = 'Chemistry',
      classifiers=[
          'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords = 'Chemistry',
      url = '..',
      license = 'MIT',
      Packages = [],
      
