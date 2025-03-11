#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File      :   illuminating_halos.py
@Time      :   2025/02
@Author    :   Matteo Calabrese
@Version   :   0.5
@Contact   :   calabrese@oavda.it
@License   :   MIT
'''

"""
illuminating_halos.py 

Dependencies:
    - numpy
    - h5py
    - astropy

Usage:
    python illuminating_halos.py parameters.ini
"""

import numpy as np

from astropy.cosmology import FlatLambdaCDM

import h5py

import sys, os


def read_parameter_file(filename):
    """
    Reads a parameter file and returns its contents as a dictionary.

    Parameters:
    filename (str): Path to the parameter file to read.

    Returns:
    dict: Dictionary containing parameters from the file.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        exit(1)

    params = {}

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Ignore lines that are comments (starting with '##' or empty lines)
                line = line.strip()
                if line.startswith('##') or not line:
                    continue

                # Split each line at the '=' sign and remove leading/trailing spaces
                key_value = line.split('=')
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()

                    # Convert value types based on its format
                    if value.lower() in ['true', 'false']:  # for boolean values
                        value = value.lower() == 'true'
                    elif value.replace('.', '', 1).isdigit() or (value[0] == '-' and value[1:].replace('.', '', 1).isdigit()):  # for float or int values
                        value = float(value) if '.' in value else int(value)

                    params[key] = value
                else:
                    print(f"Warning: Skipping malformed line: {line}")
            return params

    except Exception as e:
        print(f"Error: Failed to read or process the file. Exception: {str(e)}")
        exit(1)


def extract_model_name(base_name_str):
    """Extracts the model-specific part from the current folder name, removing 'BaseName' string."""
    
    # Get the current working directory
    current_path = os.getcwd()
    # Extract the name of the current folder (model name)
    current_folder_name = os.path.basename(current_path)
    # Remove the 'BaseName' part from the folder name
    if base_name_str in current_folder_name:
        model_name = current_folder_name.replace(base_name_str, "")
        return model_name
    else:
        print(f"Error: The folder name does not contain '{base_name_str}'.")
        sys.exit(1)
        
def check_success_file(directory, log_file):
    """Checks if a SUCCESS file exists in the specified directory. Logs and exits if not found."""
    success_path = os.path.join(directory, "SUCCESS")
    if not os.path.exists(success_path):
        with open(log_file, 'a') as log:
            log.write("Catalogue not ready\n")
        print("Catalogue not ready. Exiting.")
        sys.exit(1)

def setup_logging(log_directory):
    """Sets up the log file in the specified writing folder."""

    # Ensure the log directory exists, create it if it doesn't
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        
    log_file = os.path.join(log_directory, "run_Xluminosity.log")
    # log_file_handler = open(log_file, 'a')  # Open for appending
    return log_file


### ILLUMINATING FUNCTIONS
def BA_mass2lum_relation(mass, a=-1.36, b=1.88, c=-0.29):
    """
    This function calculates the luminosity from the mass using the 
    Balaguera-Antolinez (2012) mass-to-luminosity relation, which is a quadratic 
    function in terms of the logarithm of the mass.

    Parameters:
    mass (float): The mass (in units of 10^14 solar masses) for which luminosity is 
                  calculated.

    Returns:
    float: The luminosity corresponding to the input mass (in units of 10^44 erg/s/h²).
    """
    
    # Logarithm of the mass divided by 10^14 to standardize units
    m = np.log10(mass / 1.e14)
    
    # Calculate the luminosity based on the quadratic mass-to-luminosity relation
    l = a + b * m + c * m**2
    
    # Return the luminosity in units of 10^44 erg/s/h²
    return 10**(l)


def hmass2luminosity(mass, a=-1.36, b=1.88, c=-0.29, sigma=0.328):
    """
    This function converts a given mass into a corresponding luminosity 
    using the Balaguera-Antolinez (2012) mass-to-luminosity relation and 
    applies a log-normal distribution to the result.

    Parameters:
    mass (float or array-like): The mass for which luminosity is to be calculated.
    a (float, optional): The coefficient for the linear term in the relation. Default is -1.36.
    b (float, optional): The coefficient for the quadratic term in the relation. Default is 1.88.
    c (float, optional): The coefficient for the cubic term in the relation. Default is -0.29.
    sigma (float, optional): The standard deviation of the log-normal distribution 
                              applied to the luminosity. Default is 0.328.

    Returns:
    float or array-like: The luminosity corresponding to the input mass, 
                          sampled from a log-normal distribution.
    """
    
    # Call the Balaguera-Antolinez (2012) mass-to-luminosity relation to get the luminosity
    Lx = BA_mass2lum_relation(mass,a=a, b=b, c=c)

    # Return the luminosity sampled from a log-normal distribution
    return np.random.lognormal(mean=np.log(Lx), sigma=sigma)


def read_pinocchio_parameterfile(filename):
    """Reads a Pinocchio run parameter file properties. """
    run_params = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.split('%')[0].strip()  # Remove inline comments
                if line and not line.startswith('#'):
                    parts = line.split()
                    key = parts[0]
                    value = ' '.join(parts[1:]) if len(parts) > 1 else None
                    run_params[key] = value
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    return run_params


def write_catalogue(filename, halos_in_box, mask=None):
    """Writes the halo data dictionary to an HDF5 file.
    
    Args:
        filename (str): The path to the output HDF5 file.
        halos_in_box (dict): Dictionary containing halo data arrays.
        mask (np.array, optional): Boolean mask array to filter data.
    """
    with h5py.File(filename, 'w') as f:
        for key, data in halos_in_box.items():
            if mask is not None:
                f.create_dataset(key, data=data[mask])
            else:
                f.create_dataset(key, data=data)

                
def define_cosmology(from_file=None):

    if (from_file is None):
        run_params = {}

        ### PINOCCHIO SIMULATIONS PARAMETERS - LCDM
        run_params["Omega0"]       = 0.25    # Omega_0 (total matter)
        run_params["OmegaLambda"]  = 0.75    # Omega_Lambda
        run_params["OmegaBaryon"]  = 0.044   # Omega_b (baryonic matter)
        run_params["Hubble100"]    = 0.70    # little h
        run_params["Sigma8"]       = 0.8     # sigma8; if 0, it is computed from P(k)
        run_params["PrimordialIndex"] = 0.96 # n_s
        run_params["DEw0"] =  -1.0           # w0 of parametric dark energy equation of state
        run_params["DEwa"] =   0.0
        run_params["BoxSize"] = 1000         ## Mpc/h
        
        # Define the cosmology
        cosmology = FlatLambdaCDM(H0=run_params["Hubble100"]*100,
                                  Om0=run_params["Omega0"],Ob0=run_params["OmegaBaryon"])
        
    else:

        run_params = read_pinocchio_parameterfile(from_file)
        ## TO DO ADACT cosmology in case of DE or nu
        cosmology = FlatLambdaCDM(H0=float(run_params["Hubble100"])*100,
                                  Om0=float(run_params["Omega0"]),Ob0=float(run_params["OmegaBaryon"]))
        
    return cosmology, run_params


def set_BA_parameters(params):
    """ This function set the default parameter M-L relation, following 
    Balaguera-Antolinez (2012) paper, if these parameters are not present 
    in the params.ini file. 
    """
    params.setdefault("BA_a", -1.36)
    params.setdefault("BA_b", 1.88)
    params.setdefault("BA_c", -0.29)
    params.setdefault("BA_sigma", 0.328)
    
    return params

def main():
    """Main function to execute the script."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <parameter_file>")
        sys.exit(1)
    
    param_file = sys.argv[1]
    params = read_parameter_file(param_file)
    params = set_BA_parameters(params)
    
    model_name = extract_model_name(params['BaseName'])
    writing_folder = params['WritingPath'] +"_"+ model_name

    logfile = setup_logging(writing_folder)
    
    with open(logfile, 'a') as log:
        log.write("Starting run...\n")

    with open(logfile, 'a') as log:
        log.write(f"Extracted model name: {model_name}\n")
    
    check_success_file(os.getcwd(), logfile)
    
    filename = f"parameter_file_{model_name}"
    cosmology, _ = define_cosmology(from_file=filename)

    redshift = params["z"]
    if (params["is_pinocchio_box"]):
        if (params["output_type"] == '.out'):

            catalogue_file = os.getcwd() + f"/pinocchio.{redshift:1.4f}.{model_name}.catalog.out"
            halos_mass, x, y, z, vx, vy, vz, nparts = np.loadtxt(catalogue_file,unpack=True,usecols=(1,5,6,7,8,9,10,11))

            halos = {"x": x, "y": y, "z": z,
                     "vx": vx, "vy": vy, "vz": vz,
                     "DM_mass": halos_mass, "Nparts": nparts}
            
        else:
            # Open the HDF5 file
            catalogue_file = os.getcwd() + f"/pinocchio.{redshift:1.4f}.{model_name}.catalog.h5"

            with h5py.File(catalogue_file, "r") as f:
                
                x = f["halo_position_final"][:,0]
                y = f["halo_position_final"][:,1]
                z = f["halo_position_final"][:,2]
                vx = f["halo_velocity"][:,0]
                vy = f["halo_velocity"][:,1]
                vz = f["halo_velocity"][:,2]
                halos_mass = f["halo_mass"][:]
                nparts = f["halo_npart"][:]
                
                halos = {"x": x, "y": y, "z": z,
                         "vx": vx, "vy": vy, "vz": vz,
                         "DM_mass": halos_mass, "Nparts": nparts }

    else:
        print("Error: Pinocchio catalogues are needed")
        with open(logfile, 'a') as log:
            log.write("Error: Pinocchio catalogues are needed\n")
        sys.exit(2)

    nclusters = len(halos["DM_mass"])
    with open(logfile, 'a') as log:
        log.write(f"Total clusters: {nclusters}\n")

    ### Illuminating all clusters
    with open(logfile, 'a') as log:
        log.write("... Illuminating all DM halos\n")
        
    halos["luminosity_X"] = hmass2luminosity(halos["DM_mass"],
                                             a=params["BA_a"], b=params["BA_b"], c=params["BA_c"],
                                             sigma=params["BA_sigma"])

    ### Selecting clusters based on Nparts and L-Lim
    with open(logfile, 'a') as log:
        log.write("... Masking catalogue due to Mass and Luminosity\n")
    mask_1 = halos["Nparts"] > params["Npart_minimum"]
    mask_2 = halos["luminosity_X"] > params["L_Limit"]

    mask = mask_1 & mask_2
    with open(logfile, 'a') as log:
        log.write(f"    Total number of clusters after mask: {np.sum(mask)}\n")

    ## Writing catalogue
    writename = writing_folder + f"/clusters_Xluminosity_z_{redshift:1.4f}.h5"
    with open(logfile, 'a') as log:
        log.write(f"... Writing catalogue: {writename}\n")
        

    write_catalogue(writename,halos, mask=mask)

    with open(logfile, 'a') as log:
        log.write("done. \n")

if __name__ == "__main__":
    main()


