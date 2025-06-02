#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File      :   illuminating_halos.py
@Time      :   2025/02
@Author    :   Matteo Calabrese
@Version   :   0.6
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
from astropy.cosmology import z_at_value
import astropy.units as u

from scipy.interpolate import CubicSpline

import h5py

import sys, os


def lightcone_coords(x,y,z, xc,yc,zc, vx,vy,vz, correct_for_RSD=True,H0=69.8):
    
    """
    Convert Cartesian coordinates to spherical coordinates (RA, Dec, Radius).
        
    Parameters:
    x, y, z (array-like): Cartesian coordinates of points in Mpc/h.
    xc, yc, zc (float): Center of the box in Mpc/h. 
    vx, vy, vz (array-like): Velocities in Cartesian coordinates in km/s.
    H0 (float): Constant for radius correction. H0 in km/s/Mpc.
    
    Returns:
    tuple: Arrays of right ascension (RA in degrees),
            declination (Dec in degrees), and radius.
            Radius is corrected for Redshift-Space-Distorsion (RSD)
    """
    # Shift the coordinates relative to the center of the box
    x_shifted = x - xc
    y_shifted = y - yc
    z_shifted = z - zc
    
    # Compute the radius
    radius = np.sqrt(x_shifted**2 + y_shifted**2 + z_shifted**2)
    
    if (correct_for_RSD):
        # Compute the radial unit vector
        u_r_x = x_shifted / radius
        u_r_y = y_shifted / radius
        u_r_z = z_shifted / radius
        
        # Compute the scalar product v . u_r
        v_dot_u_r = vx * u_r_x + vy * u_r_y + vz * u_r_z
        
        # Correct the radius
        radius = radius + v_dot_u_r / H0
        
    # Compute the declination (Dec) in degrees
    with np.errstate(invalid='ignore', divide='ignore'):  # Suppress warnings temporarily
        z_safe = np.clip(z_shifted / radius, -1.0, 1.0)  # Clamp z / radius to [-1, 1]
        dec = np.arcsin(z_safe) * (180 / np.pi)  # Compute declination in degrees
        
    # Compute the right ascension (RA) in degrees
    ra = np.arctan2(y_shifted, x_shifted) * (180 / np.pi)
    ra = np.mod(ra, 360)  # Ensure RA is in the range [0, 360)
        
    return ra, dec, radius


def distance_to_redshift(distance_mpc_h,cosmology):
    """
    Calculate the redshift for a given comoving distance in Mpc/h using the specified cosmology.

    Parameters:
    -----------
    distance_mpc_h : float
        Comoving distance in Mpc/h.
    cosmology :  object
        Cosmology object from astropy
    
    Returns:
    --------
    redshift : float
        The redshift corresponding to the input comoving distance.
    """
    h = cosmology.H0.value / 100
    
    # Convert distance from Mpc/h to Mpc
    distance_mpc = distance_mpc_h / h

    # Compute the redshift
    redshift = z_at_value(cosmology.comoving_distance, distance_mpc * u.Mpc)
    
    return redshift

def distance_to_redshift_spline(cosmology, r_start=0.01, r_end=500):
    """ make a spline to make it fast """
    radii = np.logspace(np.log10(r_start),np.log10(r_end),1000)
    redshifts = [distance_to_redshift(r,cosmology) for r in radii]
    return CubicSpline(radii, redshifts)


def Lx_limits_function(Lmin_z_experiment_file,z):
    
    data = np.genfromtxt(Lmin_z_experiment_file, delimiter=',', skip_header=1)
    z_l = data[:,0]
    log10_L_lim = data[:,1]
    
    coefficients = np.polyfit(z_l, log10_L_lim, 4)  # 2 for quadratic fit
    quadratic_function = np.poly1d(coefficients)

    return quadratic_function(z)
    
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
        
def check_success_file(directory, log_file, use_abacus=False):

    if (use_abacus):
        return 0;
    """Checks if a SUCCESS file exists in the specified directory. Logs and exits if not found."""
    success_path = os.path.join(directory, "SUCCESS")
    if not os.path.exists(success_path):
        with open(log_file, 'a') as log:
            log.write("Catalogue not ready\n")
        print("Catalogue not ready. Exiting.")
        sys.exit(1)

def setup_logging(log_directory,fname='std'):
    """Sets up the log file in the specified writing folder."""

    # Ensure the log directory exists, create it if it doesn't
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        
    log_file = os.path.join(log_directory, f"run_Xluminosity.{fname}.log")
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


def hmass2luminosity(mass, a=-1.36, b=1.88, c=-0.29, sigma=0.328, seed=-1):
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


def hmass2CountRate(mass, a=-1.36, b=1.88, c=-0.29, sigma=0.328, seed=-1):
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



def load_model_samples(npz_path, model_key):
    # Load the .npz file
    data = np.load(npz_path)

    if model_key not in data:
        raise ValueError(f"Model key '{model_key}' not found in the .npz file.")

    samples = data[model_key]  # shape: (n_samples, 4)

    # Split into 4 arrays, one per parameter
    param_arrays = [samples[:, i] for i in range(samples.shape[1])]

    return param_arrays  # List of 4 arrays, each of shape (n_samples,)


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


def write_catalogue(filename, halos_in_box, mask=None, is_lightcone=False):
    """Writes the halo data dictionary to an HDF5 file.
    
    Args:
        filename (str): The path to the output HDF5 file.
        halos_in_box (dict): Dictionary containing halo data arrays.
        mask (np.array, optional): Boolean mask array to filter data.
    """
    if (is_lightcone):
        halos = {}
        halos["x"] = halos_in_box["x"]
        halos["y"] = halos_in_box["y"]
        halos["z"] = halos_in_box["z"]
        halos["ra"] = halos_in_box["ra"]
        halos["dec"] = halos_in_box["dec"]
        halos["redshift_h"] = halos_in_box["redshift_h"]
        halos["DM_mass"] = halos_in_box["DM_mass"]
        halos["luminosity_X"] = halos_in_box["luminosity_X"]
    else:
        halos = halos_in_box
    
    with h5py.File(filename, 'w') as f:
        for key, data in halos_in_box.items():
            if mask is not None:
                f.create_dataset(key, data=data[mask])
            else:
                f.create_dataset(key, data=data)

                
def define_cosmology(from_file=None, from_abacus=False):

    if (from_file is None or from_abacus is True):
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

    if (params["seed"] != -1):
        # Fix the seed for reproducibility
        np.random.seed(params["seed"])
    
    BaseName = params['BaseName']
    model_name = extract_model_name(params['BaseName'])
    writing_folder = params['WritingPath'] +"_"+ model_name
    redshift = params["z"]
    mstring = params["NameConv"]
    
    logfile = setup_logging(writing_folder,fname=f'{model_name}.{redshift:1.4f}.{mstring}')
    
    with open(logfile, 'a') as log:
        log.write("Starting run...\n")

    with open(logfile, 'a') as log:
        log.write(f"Extracted model name: {model_name}\n")
    
    check_success_file(os.getcwd(), logfile,use_abacus=params["use_abacus"])
    
    filename = f"parameter_file_{model_name}"
    cosmology, run_params = define_cosmology(from_file=filename,from_abacus=params["use_abacus"])

    n_redshifts = 1
    if (n_redshifts > 1):
        print("full lightcone")
    else:
        if (params["is_pinocchio_box"]):
            if (params["output_type"] == '.out'):
                
                catalogue_file = os.getcwd() + f"/pinocchio.{redshift:1.4f}.example.catalog.out"
                halos_mass, x, y, z, vx, vy, vz, nparts = np.loadtxt(catalogue_file,unpack=True,usecols=(1,5,6,7,8,9,10,11))
                
                halos = {"x": x, "y": y, "z": z,
                         "vx": vx, "vy": vy, "vz": vz,
                         "DM_mass": halos_mass, "Nparts": nparts}
                
            else:
                # Open the HDF5 file
                catalogue_file = os.getcwd() + f"/pinocchio.{redshift:1.4f}.{model_name}.catalog.h5"
                if (params["use_abacus"]):
                    catalogue_file = os.getcwd() + f"/{BaseName}{model_name}_halos_z{redshift:1.4f}.h5"
                    with open(logfile, 'a') as log:
                        log.write("Reading abacus halos in {catalogue_file}\n")
                        
                    with h5py.File(catalogue_file, "r") as f:
                        ### tags: "halo_mass', 'halo_npart', 'halo_position', 'halo_velocity'
                        x = f["halo_position"][:,0]
                        y = f["halo_position"][:,1]
                        z = f["halo_position"][:,2]
                        vx = f["halo_velocity"][:,0]
                        vy = f["halo_velocity"][:,1]
                        vz = f["halo_velocity"][:,2]
                        halos_mass = f["halo_mass"][:]
                        nparts = f["halo_npart"][:]
                    
                        halos = {"x": x, "y": y, "z": z,
                                 "vx": vx, "vy": vy, "vz": vz,
                                 "DM_mass": halos_mass, "Nparts": nparts }
                        
                else:
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

    n_run_BA_parameters = 0
    if ("BA_parameters_run" in params):
        with open(logfile, 'a') as log:
            log.write(f"... Doing a Sobol run on BA L-M paramters ...\n")

        model_key = model_name
        if (params["use_abacus"]):
            model_key = BaseName + model_name
            
        BA_params = load_model_samples(params["BA_parameters_run"], model_key)
        
        n_run_BA_parameters = len(BA_params[0])
        
        with open(logfile, 'a') as log:
            log.write(f"... First run is with standard values, then {n_run_BA_parameters} realizations ...\n")
    
    if (params["is_output_lightcone"]):
        
        with open(logfile, 'a') as log:
            log.write("... Illuminating DM halos in lightcone, single box\n")

        halos["luminosity_X"] = hmass2luminosity(halos["DM_mass"],
                                                 a=params["BA_a"], b=params["BA_b"], c=params["BA_c"],
                                                 sigma=params["BA_sigma"])


        BoxSize = float(run_params["BoxSize"])
        #centre of mass
        xc,yc,zc = (BoxSize/2,BoxSize/2,BoxSize/2)
        
        
        ra, dec, radius = lightcone_coords(halos["x"],halos["y"],halos["z"], xc,yc,zc,
                                           halos["vx"],halos["vy"],halos["vz"],
                                           H0=cosmology.H0.value)

        halos["ra"] = ra
        halos["dec"] = dec
        
        MaxRadius = BoxSize/2 #Mpc/h

        with open(logfile, 'a') as log:
            log.write(f"... Lightcone geometry: centre({xc},{yc},{zc}) radius: {MaxRadius}\n")
        
        redshift_max = distance_to_redshift(MaxRadius,cosmology)

        mask_1 = radius < MaxRadius

        with open(logfile, 'a') as log:
            log.write(f"... : Lightcone, selecting halos in z=[0,{redshift_max:.2f}]\n")

        dist2red = distance_to_redshift_spline(cosmology,
                                               r_start=0.01, r_end=BoxSize/2*np.sqrt(3))
        
        halos["redshift_h"] = dist2red(radius)
        
        Lmin_z_experiment = params["Lmin_z_experiment"]
        Lx_lims = Lx_limits_function(Lmin_z_experiment,halos["redshift_h"])

        mask_2 = (halos["luminosity_X"] > Lx_lims) & (halos["luminosity_X"] > params["L_Limit"])
        # mask_2 = halos["luminosity_X"] > Lx_lims

        with open(logfile, 'a') as log:
            log.write(f"... : Lightcone, selecting halos using {Lmin_z_experiment} Lmin(z)\n")
            
        mask = mask_1 & mask_2
        # mask = mask_1
        
        with open(logfile, 'a') as log:
            log.write(f"    Total number of clusters after masking: {np.sum(mask)}\n")

        ## Writing catalogue
        writename = writing_folder + f"/clusters_Xluminosity_lightcone_single_z_{redshift:1.4f}.h5"
        with open(logfile, 'a') as log:
            log.write(f"... Writing catalogue: {writename}\n")
        
        write_catalogue(writename,halos, mask=mask,is_lightcone=True)
            
    else:

        ### Illuminating all clusters
        with open(logfile, 'a') as log:
            log.write("... Illuminating all DM halos\n")
            OA
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

    for n in range(n_run_BA_parameters):

        BA_a = BA_params[0][n]
        BA_b = BA_params[1][n]
        BA_c = BA_params[2][n]
        BA_scatter = BA_params[3][n]
        
        if (params["is_output_lightcone"]):
            
            with open(logfile, 'a') as log:
                log.write("... Illuminating DM halos in lightcone, single box\n")

            halos["luminosity_X"] = hmass2luminosity(halos["DM_mass"],
                                                     a=BA_a, b=BA_b, c=BA_c,
                                                     sigma=BA_scatter)


            BoxSize = float(run_params["BoxSize"])
            #centre of mass
            xc,yc,zc = (BoxSize/2,BoxSize/2,BoxSize/2)
                
            ra, dec, radius = lightcone_coords(halos["x"],halos["y"],halos["z"], xc,yc,zc,
                                               halos["vx"],halos["vy"],halos["vz"],
                                               H0=cosmology.H0.value)

            halos["ra"] = ra
            halos["dec"] = dec
        
            MaxRadius = BoxSize/2 #Mpc/h
            # with open(logfile, 'a') as log:
            #     log.write(f"... Lightcone geometry: centre({xc},{yc},{zc}) radius: {MaxRadius}\n")
        
            redshift_max = distance_to_redshift(MaxRadius,cosmology)

            mask_1 = radius < MaxRadius

            # with open(logfile, 'a') as log:
            #     log.write(f"... : Lightcone, selecting halos in z=[0,{redshift_max:.2f}]\n")

            dist2red = distance_to_redshift_spline(cosmology,
                                                   r_start=0.01, r_end=BoxSize/2*np.sqrt(3))
        
            halos["redshift_h"] = dist2red(radius)
        
            Lmin_z_experiment = params["Lmin_z_experiment"]
            Lx_lims = Lx_limits_function(Lmin_z_experiment,halos["redshift_h"])

            mask_2 = (halos["luminosity_X"] > Lx_lims) & (halos["luminosity_X"] > params["L_Limit"])

            with open(logfile, 'a') as log:
                log.write(f"... {n} Lightcone, selecting halos using {Lmin_z_experiment} Lmin(z)\n")
            
            mask = mask_1 & mask_2
        
            with open(logfile, 'a') as log:
                log.write(f"   {n} Total number of clusters after masking: {np.sum(mask)}\n")

                ## Writing catalogue
            writename = writing_folder + f"/clusters_Xluminosity_lightcone_{n}_single_z_{redshift:1.4f}.h5"
            with open(logfile, 'a') as log:
                log.write(f"... Writing catalogue: {writename}\n")
        
            write_catalogue(writename,halos, mask=mask,is_lightcone=True)
            
        else:

            ### Illuminating all clusters
            with open(logfile, 'a') as log:
                log.write("... Illuminating all DM halos\n")
            
            halos["luminosity_X"] = hmass2luminosity(halos["DM_mass"],
                                                     a=BA_a, b=BA_b, c=BA_c,
                                                     sigma=BA_scatter)
            
            ### Selecting clusters based on Nparts and L-Lim
            # with open(logfile, 'a') as log:
            #     log.write("... Masking catalogue due to Mass and Luminosity\n")
                
            mask_1 = halos["Nparts"] > params["Npart_minimum"]
            mask_2 = halos["luminosity_X"] > params["L_Limit"]

            mask = mask_1 & mask_2
            with open(logfile, 'a') as log:
                log.write(f"  {n}  Total number of clusters after mask: {np.sum(mask)}\n")

                ## Writing catalogue
            writename = writing_folder + f"/clusters_Xluminosity_{n}_z_{redshift:1.4f}.h5"
            with open(logfile, 'a') as log:
                log.write(f"... Writing catalogue: {writename}\n")
        
            write_catalogue(writename,halos, mask=mask)

    with open(logfile, 'a') as log:
        log.write("done. \n")

if __name__ == "__main__":
    main()


