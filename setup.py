import setuptools

"""
https://packaging.python.org/tutorials/packaging-projects/
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mederrata-spmf",
    version="0.0.1",
    author="mederrata",
    author_email="info@mederrata.com",
    description="mederrata sparse poisson matrix factorization tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mederrata/spmf",
    packages=setuptools.find_packages(
        exclude=["*.md", "aws", "bin/*.sh", "design_docs", "tools/"]
    ),
    include_package_data=True,
    package_data={
        "mederrata_spmf": [
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Closed",
        "Operating System :: Linux",
    ],
    scripts=[
        "bin/factorize_csv.py"
    ],
    python_requires='>=3.6',
    install_requires=[
        'dill>=0.3.1.1',
        'matplotlib>=3.1',
        'arviz>=0.10.0',
        'numpy>=1.17',
        'pandas>=1.0.0, <1.2.0',
        # We need to check for direct depends or we can delete scipy.
        'scipy>=1.4.1',
        'tensorflow>=2.4.0',
        'tensorflow-probability>=0.12.1',
        'tensorflow-addons>=0.12.0',
        'bayesianquilts@git+https://github.com/mederrata/bayesianquilts.git'
    ],
    dependency_links=[
        "git+https://github.com/mederrata/bayesianquilts.git",
    ]
)
