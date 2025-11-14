Pipeline applications
============

Introduction
-----------
We provide scripts to authomatize individual step of the COSI data analysis.
These can be called from the command line and require, at minimum, only a .yaml configuration file listing the input parameters.
and files (cosi dataset, response file, orientation file, background file) to process.
Here, we demonstrate the usage of the available applications, which are, at the moment:

cosi-bindata: to bin an unbinned dataset, matching a given response matrix
cosi-threemlfit: to fit a source at l,b with a given threeml spectral model

For all the applications, the option -h or -h --help prints an help file that describes the function, the inputs, the outputs and the command line option available

cosi-threemlfit --help

The .yaml parameter file
~~~~~~~~~~~~

How to call the applications from the command line
~~~~~~~~~~~~
A .yaml parameters file is the minimum requirement to call an application e.g.:

cosi-bindata --config_file MY_CONFIG_FILE

It is possible to add options directly in the command line e.g. :

cosi-bindata --config_file MY_CONFIG_FILE --odir MY_OUTPUT_DIRECTORY

This is the same as before, but the outputs will be saved in the given ouput directory.

It is also possible to override parameters of the .yaml file, either directly (for same parameters) or using the override option for any parameters.
For instance:
cosi-bindata --config_file MY_CONFIG_FILE --tmin TMIN --tmax TMAX

this string will run cosi-bindata using the tmin amd tmax provided in the command line
or
cosi-threemlfit --config_file MY_CONFIG_FILE --override cuts:kwargs:tstart=TSTART cuts:kwargs:tstop=TSTOP

will run

How to call the applications from a python script
~~~~~~~~~~~~


# run_bindata.py
from bindata_script import cosi_bindata# Define the arguments as a list of strings
# This mimics what you'd type on the command line
arguments = [
    '--config', 'config.yaml',      # Required config file
    '--tmin', '1577836800.0',       # Optional tmin override
    '--tmax', '1577923200.0',       # Optional tmax override
    '-o', './output_directory',     # Specify the output directory
    '--suffix', 'my_run',           # Add a suffix to output files
    '--overwrite'                   # Flag to overwrite existing files
]# Call the function with the argument list
# The function will use this list instead of sys.argv
cosi_bindata(argv=arguments)print("cosi_bindata function executed successfully.")





A practical example
~~~~~~~~~~~~

