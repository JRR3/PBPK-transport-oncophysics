# PBPK-transport-oncophysics

  (1) Origin
---------------------------------------------------------------------- 
This implementation was coded at the Houston Methodist Research Institute. It is
part of a project coordinated by the PI Mauro Ferrari, supervised by Arturas
Ziemys and developed by Javier Ruiz Ramirez.

  (2) Summary
---------------------------------------------------------------------- 
This is a Python implementation of a physiologically-based pharmacokinetic
(PBPK) model intended to be used in the design and simulation of immune
therapies. The focus is on the transport of cells and antigen across different
tissues. 

  (3) Python files
----------------------------------------------------------------------
The code is composed of the following python files:

* driver.py

Coordinates all modules and executes the program

* pbpk_l_module.py

PBPK management, load material properties, and build main components. 

* compartment_module.py  

Definition of compartments, i.e., the black boxes that describe the various
physiological structures and its behaviours.

* species_module.py

Management of all the material and transport properties of the various
compartments.

* mouse_parameter_module.py  

A small list of physiological parameters relevant to the mouse model.

* tumor_module.py

Defines the tumor attributes and behaviours.

* postprocessing_module.py

Process the data and generate plots

* plotting_instructions_module.py  

Define plotting instructions for graphical output.


  (4) CSV files
----------------------------------------------------------------------
These files are separated in two folders

  (5) Execution
----------------------------------------------------------------------
One simply has to run the command 

      python driver.py


