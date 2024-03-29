# Protein-MGEM

## Hardware Requirements

* For re-training BERT models, at least a single V100 NVIDIA GPU (40 GB VRAM)
is needed, though 2 are recommended.
* For evaluation and analysis, any workstation running a UNIX system may be used. 
For acceptable performance, at least 16 CPUs and 32 GB of RAM are recommended.


## Software Requirements

1. The dependencies required for the Protein-MGEM scripts are given through
the conda environment specifications `protein_mgem.yaml`.
An environment created from this must be active to run the scripts.

2. Additionally, the Protein-MGEM code base should be installed into the environment
by running `pip install -e protein_mgem` after cloning this repo.

As an alternative to 1., a Singularity container that encapsulates the required
conda environment may be used to run the scripts.
A pre-built container may be pulled from the Sylabs Cloud Library with the command:
`singularity pull protein_mgem.sif library://fburic/protein-mgem/protein_mgem:latest`

The container is expected to be found on the repo's top directory level.

The user may also build their own container from the supplied `protein_mgem.def` file
(requires root permissions, 
see instructions [here](https://sylabs.io/guides/3.0/user-guide/build_a_container.html)).

### Operating system

Protein-MGEM was developed and used on POSIX systems: 
Ubuntu 18.04, CentOS 7.8 Linux, as well as macOS 10.15 (Catalina) - 12.5.1 (Monterey)


## Models and Data

These may be downloaded from Zenodo by running the respective download scripts:
`download_models.sh` and `download_data.sh`

Intermediate large results (such as attention profiles) are stored in their corresponding
experiment directory, while initial inputs are kept in `data`.


## Running Scripts

The various scripts can be run via a Python interpreter directly (i.e. `python SCRIPT ARGS`)
or by using the Singularity wrapper (`run_python.sh SCRIPT ARGS`).
The latter calls the Python interpreter in the `protein_megem.sif` Singularity image
(either downloaded or built by the user - see Software Requirements).

For the various scripts and parameters, see the [research log](results/20211223_182228/README.md)
As a general rule, the scripts use the information supplied in `experiment_config.yaml`
(input files and certain parameter values) as input arguments.


## Unit Tests

Some functions in the codebase have unit tests for assessing their correctness.
These may also be used for a quick check that the codebase is runnable.

To run these, from the repo top directory level execute: `./test/run_unit_tests.sh`

