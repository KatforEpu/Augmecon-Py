*** ---------------------------------------------------------------------------- ***
*** INSTALLATION OF THE AUGMECON-PY SOFTWARE ***
*** ---------------------------------------------------------------------------- ***

For the successful installation of the Augmecon-Py code the user must perform the following actions:


1. Download & install the latest python version
----------------------------------------------

The latest version of the python programming language can be accessed from the following link: https://www.python.org/downloads/


2. Installation of the Gurobi optimizer (or any similar optimizer for example CPLEX) and obtain the relevant license (the authors use the academic license of gurobi, while purchase of a commercial license in also possible)
---------------------------------------------------------------------------
   
The developed Augmecon-Py software requires an active installation of the Gurobi optimizer and a license for the research and educational utilization of the gurobi solver.
a Gurobi license for academics and researchers can be found here: https://www.gurobi.com/academia/academic-program-and-licenses/


3. Install required python packages
----------------------------------

After the succesful installation of the python programming environment the user should install several packages that are required for the execution of the optimization code. All required packages are included within the "requirements.txt" file and the code for installing them through the terminal is provided below:

python -m pip install -r requirements.txt


5. Creation of required folders
-------------------------------

Finally, the user should create the following basic folders in the path of the code

"input"
"output" 


