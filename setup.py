import setuptools

"""
https://packaging.python.org/tutorials/packaging-projects/
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mederrata-spmf",
    version="0.0.4",
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
        'bayesianquilts@git+https://github.com/mederrata/bayesianquilts.git'
    ],
    dependency_links=[
        "git+https://github.com/mederrata/bayesianquilts.git",
    ]
)
