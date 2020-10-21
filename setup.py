# import sys
# import versioneer
from setuptools import setup


# if {'build_sphinx'}.intersection(sys.argv):
#     setup_requires.append('sphinx')
# if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
#     setup_requires.append('pytest-runner')


setup(
    setup_requires=[
        'setuptools >= 30.3.0',
        'setuptools_scm',
    ],
    use_scm_version={
        'write_to': 'gwinc/_version.py',
    }
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
)
