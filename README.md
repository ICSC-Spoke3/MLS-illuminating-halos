# Illuminating Halos

## Overview
Illuminating Halos is a Python script designed to process and analyze halo data from cosmological simulations. The script reads halo catalogs generated with [Pinocchio](https://adlibitum.oats.inaf.it/pierluigi.monaco/pinocchio.html) and applies the Mass to Luminosity approach developed by [Balaguera-Antolinez (2012)](https://arxiv.org/abs/1207.2138) to illuminate dark matter halos with X-ray luminosity. The script reads input parameters from a configuration file and utilizes scientific libraries to handle large datasets efficiently.


## Features
- Reads and processes halo data from HDF5 files.
- Computes and analyzes halo properties.
- Supports parameterized configuration via an external `.ini` file.
- Leverages optimized scientific libraries for efficient computations.

## Dependencies
To run `illuminating_halos.py`, you need the following Python libraries:

- [numpy](https://numpy.org/) (for numerical computations)
- [h5py](https://www.h5py.org/) (for handling HDF5 files)
- [astropy](https://www.astropy.org/) (for astrophysical calculations)

Ensure you have them installed by running:
```bash
pip install numpy h5py astropy
```

## Installation
Clone the repository and navigate to the directory:
```bash
git clone https://github.com/your-repo/illuminating_halos.git
cd illuminating_halos
```

## Usage
To run the script, provide a configuration file as input:
```bash
python illuminating_halos.py parameters.ini
```
### Configuration File (`parameters.ini`)

The script reads its configuration parameters from the `parameters.ini` file. Ensure that this file contains the required settings in the correct format.

A sample `parameters.ini` file is shown below:

```ini
[General]
BaseName          = L1500_N750_sobol_ndim2_
WritingPath       = /home/path/to/TestXLF/L1500_N750_sobol_ndim2
is_pinocchio_box  = True
output_type       = .h5
z                 = 0.0000

[SimulationSettings]
Npart_minimum     = 40
L_Limit           = 0.003
```

### Parameters Description:

- **BaseName**: The base name used for simulations or data sets.
- **WritingPath**: The directory where the output files will be stored.
- **is_pinocchio_box**: A boolean flag to indicate whether to use the Pinocchio box or lightcone (set as `True` or `False`).
- **output_type**: Specifies the type of the output file (e.g., `.h5`).
- **z**: The redshift value (can be adjusted according to the needs of the simulation).
- **Npart_minimum**: The minimum number of particles for the cluster to be illuminated.
- **L_Limit**: A parameter defining the luminosity limit for the survey (e.g., 0.003 10^44 erg/s/h2 for REFLEX-II).

Ensure that these parameters are correctly set in your `parameters.ini` file before running the script.

## Example
Here’s an example of running the script:
```bash
python illuminating_halos.py config.ini
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Feel free to submit issues or pull requests to improve the script.

## Contact
For questions or feedback, contact [calabrese@oavda.it](mailto:calabrese@oavda.it) or open an issue on GitHub.
