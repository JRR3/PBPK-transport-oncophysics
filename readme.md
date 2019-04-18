# PBPK-transport-oncophysics

## Origin
---------------------------------------------------------------------- 
This implementation was coded at the [Houston Methodist Research Institute]
(https://www.houstonmethodist.org/research/).
It is part of a project coordinated by the PI Mauro Ferrari,
supervised by Arturas Ziemys and developed by Javier Ruiz Ramirez.

##  Summary
---------------------------------------------------------------------- 
This is a Python(2.7.7) implementation of a physiologically-based pharmacokinetic
(PBPK) model intended to be used in the design and simulation of immune
therapies. The focus is on the transport of cells and antigens across different
tissues. The principal route of transport is the lymphatic system.

##  Libraries
----------------------------------------------------------------------
The following libraries are used in the execution of this program. 
Make sure you have installed these.

1. NumPy
 * Scientific computing

2. matplotlib
 * Plot generation

3. pandas
 * Handle data structures and data analysis

##  Modules
----------------------------------------------------------------------
The implementation is composed of the following modules:

1. ```driver.py```
 * Coordinates all modules and executes the program

2. ```pbpk_l_module.py```
 * PBPK management, load material properties, and build main components

3. ```compartment_module.py```
 * Definition of compartments, i.e., the black boxes that describe the various
physiological structures and its behaviors.

4. ```species_module.py```
 * Management of all the material and transport properties of the various
compartments.

5. ```mouse_parameter_module.py```  
 * A small list of physiological parameters relevant to the mouse model.

6. ```tumor_module.py```
 * Defines the tumor attributes and behaviors.

7. ```postprocessing_module.py```
 * Process the data and generate plots

8. ```plotting_instructions_module.py```  
 * Define plotting instructions for graphical output.

## CSV files
----------------------------------------------------------------------
These files are separated into 3 folders

1. pbpk_properties
2. species_properties
3. tumor_properties

## Execution
----------------------------------------------------------------------
1. Download the compressed file and extract it in your Documents folder
2. Open your console and run the command

```
      python driver.py
```

## Graphical output
----------------------------------------------------------------------
All output is generated in the graphical_output folder

## Numerical results
----------------------------------------------------------------------
After solving the system of ODEs, the solution vectors are stored in the
numerical_output folder.

