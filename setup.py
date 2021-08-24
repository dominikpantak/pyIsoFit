import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pyIsoFit",                     # This is the name of the package
    version="0.0.2",                        # The initial release version
    author="Dominik Pantak",                     # Full name of the author
    description="Package for The Fitting of Adsorption Isotherms and Prediction of Multi-component Isotherms",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages('src'),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    package_dir={'': 'src'},
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=required          # Install other dependencies if any
)