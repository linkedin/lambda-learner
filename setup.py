from os import path
from setuptools import find_namespace_packages, setup
this_directory = path.abspath(path.dirname(__file__))
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lambda-learner',
    namespace_packages=['linkedin'],
    version='0.1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['Programming Language :: Python :: 3',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved'],
    license='BSD-2-CLAUSE',
    keywords='lambda-learner incremental training',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src', exclude=['test*', 'doc']),
    url='https://github.com/linkedin/lambda-learner',
    project_urls={
        'Documentation': 'https://github.com/linkedin/lambda-learner/blob/main/README.md',
        'Source': 'https://github.com/linkedin/lambda-learner',
        'Tracker': 'https://github.com/linkedin/lambda-learner/issues',
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1.19.4',
        'scipy >= 1.5.4',
        'scikit-learn >= 0.24.0',
        'typing-extensions >= 3.7.4',
    ],
    tests_require=[
        'pytest',
    ]
)
