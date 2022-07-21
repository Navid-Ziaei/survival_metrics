from setuptools import setup
import unittest


def readme():
    with open('README.rst') as f:
        return f.read()


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir='tests',
        pattern='test_*.py',
        top_level_dir='survival_metrics')
    return test_suite


setup(name='survival_metrics',
      version='0.1',
      description='Metrics for evaluating survival analysis models',
      long_description=readme(),
      url='http://github.com/Navid-Ziaei/survival_metrics',
      author='Navid Ziaei',
      author_email='n2ziaee@gmail.com',
      license='MIT',
      packages=['survival_metrics'],
      install_requires=[
          'numpy',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='setup.my_test_suite',
      )
