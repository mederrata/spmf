import setuptools

"""
https://packaging.python.org/tutorials/packaging-projects/
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mederrata_spmf",  # Replace with your own username
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
    python_requires='>=3.6',
    install_requires=[
        'boto3',
        'datashader>=0.9.0',
        'dill>=0.3.1.1',
        'enum34;python_version<"3.4"',  # TODO: check if it removes enum34
        'gpustat',
        'importlib_resources;python_version<"3.7"',  # Standard Library 3.7+
        'jax>=0.1.5',
        'matplotlib>=3.1',
        'notebook>=6.0.0',
        'numpy>=1.17',
        'pandas>=0.3',
        'scipy==1.4.1',  # Tensorfow 2.1 depends
        'SQLAlchemy>=1.3.12',
        'tensorflow>=2.2.0rc1, < 2.3.0',
        'tensorflow-probability>=0.9',
        'tensorflow-addons>=0.7.1',
        'umap-learn>=0.3.10'
    ]
)
