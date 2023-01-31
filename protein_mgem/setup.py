import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Protein-MGEM",
    version="dev",
    author="Filip Buric",
    author_email="",
    description="Protein-MGEM scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 10.0",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.7',
)
