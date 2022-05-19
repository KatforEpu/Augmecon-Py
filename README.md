Augmecon-Py: A Python framework for multi-objective linear optimisation under uncertainty
=========================================================================================

[![license](https://img.shields.io/badge/License-Apache%202.0-black)](https://github.com/IAMconsortium/pyam/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-_3.8_|_3.9_|_3.10-blue?logo=python&logoColor=white)](https://github.com/IAMconsortium/pyam)


Installation of the Augmecon-Py software
----------------------------------------

### Install the latest Python version

The latest version of Python can be found in the following link: https://www.python.org/downloads/. A Python version 3.8 or higher is recommended.


### Installation of the optimizer 

Augmecon-Py requires an installation of the Gurobi optimizer (or any similar optimizer, for example, CPLEX). A Gurobi license for academics and researchers can be found here: https://www.gurobi.com/academia/academic-program-and-licenses/. Additionally, a Gurobi version of 9.5.1 or higher is recommended.


### Install required Python packages

After the succesful installation of Python, the user should install several packages that are required for the execution of the optimization code. All required packages are included within the "requirements.txt" file. The packages can be installed through pip using the following command on the terminal:
```
python -m pip install -r requirements.txt
```

Hardware requirements
----------------------------------------

Minimum requirements:
- CPU: Intel Core i5 @ 1.20GHz
- RAM: 8 GB

Recommended requirements (for executing the code in reasonable time):
- CPU: Intel Core i7 @ 1.30GHz
- RAM: 12 GB
