###############################################
#This project originated at the:
#Source    : Houston Methodist Research Institute
#Location  : Houston, TX
#PI        : Mauro Ferrari
#Supervisor: Arturas Ziemys
#Developer : Javier Ruiz Ramirez
#Origin    : August 2017
###############################################

from scipy.optimize import leastsq
import os
import numpy as np
import pandas as pd

'''
This module contains the data to populate the physiological and
transport properties related to the mouse model.
'''

'''Units are in mm'''
mesentery_capillary_diameter = 6.5 / 1000.

'''
Tumor apparent diffusion coefficient as an
imaging biomarker to predict tumor
aggressiveness in patients with estrogen-
receptor-positive breast cancer
NMR in Biomedicine June 2016

Units are in mm^2/s
'''
apparent_diffusion_coefficient_in_tumor = 0.714e-3

'''Units are in mm^2/day'''
apparent_diffusion_coefficient_in_tumor *= 60 * 60 * 24


'''
Tumor-Infiltrating FOXP3 T Regulatory Cells Show
Strong Prognostic Significance in Colorectal Cancer

JOURNAL OF CLINICAL ONCOLOGY January 2009

Units are in cells/ mm^2
'''

density_of_cd8_cells_in_colorectal_tumor_tissue = 147


'''
Dimensions of Blood Vessels from Distributing
Artery to Collecting Vein
Mary P. Wiedeman, Ph.D.
'''

'''Units are in mm'''
mesentery_capillary_radius = mesentery_capillary_diameter / 2.

'''
Insert reference
'''

'''Units are in mm'''
hev_wall_thickness = 0.32 / 1000.


'''Units are in micro liters'''
lymph_node_sample_tissue_volume = 400.

'''
Insert reference
'''
'''Units are in micro liters'''
spleen_volume = 180.

'''
Acquisition of full effector function in vitro
paradoxically impairs the in vivo antitumor
efficacy of adoptively transferred CD8 + T cells

These data correspond to the growth of a melanoma tumor using early effector 
CD8+ cells.
The measurements correspond to the product of the perpendicular diameters.
X: Time (days)
Y: Area (mm^2)
path       = r'../papers_for_3rd_example'
fname      = r'tumor_growth.csv'
fname      = os.path.join(path, fname)
tumor_area_data = pd.read_csv(fname, header = None, index_col = False)
tumor_area_data = tumor_area_data.values
'''


'''
Compute tumor average radius and volume from previously loaded area data
Length units are in mm
tumor_area_data[tumor_area_data < 0] = 0.
tumor_average_area     = np.mean(tumor_area_data[:,1])
tumor_average_diameter = np.sqrt(tumor_average_area)
tumor_average_radius = tumor_average_diameter * 0.5
tumor_average_volume = 4. / 3. * np.pi * np.power(tumor_average_radius, 3)
'''
#print 'TAA:', tumor_average_area
#print 'TAR:', tumor_average_radius
#print 'TAV:', tumor_average_volume
#exit()

#------------------------------------------------------------
#------------------------------------------------------------
def get_volume_of_generic_lymph_node_from_txt(self):
    '''
    Compute the volume of lymphoid tissue used in the Circadian paper.
    The units are in micrometers.
    '''

    path  = r'/home/rjavier/Documents/pbpk/imm_diag/' +\
            r'point_data_for_examples/'

    fname = r'tissue_lymph_nodes/tissue_dimensions.csv'

    full_fname = path + fname
    '''
    The units are in micrometers
    '''
    frame = pd.read_csv(full_fname, index_col=False, header=None)
    '''
    Convert micrometers to mm
    '''
    values = frame.values / 1e3

    length = np.linalg.norm(values[0] - values[1])
    width  = np.linalg.norm(values[2] - values[3])
    '''
    Assume the volume of the lymphoid tissue is tumor-like
    See additional references for the usage of this formula.
    '''
    volume = 0.5 * length * (width**2)


#------------------------------------------------------------
#------------------------------------------------------------
def fit_data_to_sinusoidal(self, path, fname):
    '''
    Get the data stored in the file fname and fit it to a sinusoidal.
    '''

    regex     = re.compile(r'[a-zA-Z0-9_]+')
    obj       = regex.match(fname)
    pure_name = obj.group(0)

    fig = plt.figure()
    ax  = fig.add_subplot(111) 

    full_fname = os.path.join(path, fname)
    print full_fname
    frame = pd.read_csv(full_fname, index_col=False, header=None)
    values = frame.values
    x_data = values[:,0]
    y_data = values[:,1]

    f_param_x =\
            lambda param, x:\
            param[0]*np.sin(2*np.pi * param[1] * x + param[2]) + param[3]

    optimize_func = lambda param: f_param_x(param, x_data) - y_data

    amplitude_guess = 0.5 * (np.max(y_data) - np.min(y_data))
    '''
    We guess the data has a period of 24 hours.
    '''
    frequency_guess = 1. / 24
    phase_guess     = 0
    shift_guess     = np.mean(y_data)
    guess           = np.array([amplitude_guess,\
            frequency_guess, phase_guess, shift_guess])
    obj = leastsq(optimize_func, guess)

    parameters = obj[0]
    #print 'Guess     :', guess
    #print 'Parameters:', parameters
    xx = np.linspace(np.min(x_data), np.max(x_data), 101)
    yy = f_param_x(parameters, xx)

    ax.plot(x_data, y_data, 'ko')
    ax.plot(xx, yy, 'b-')

    
    fig.suptitle(pure_name)
    fig_name      = pure_name + '.pdf'
    full_fig_name = os.path.join(path, fig_name)

    fig.savefig(full_fig_name, format = 'pdf')

    '''
    Units are 10^4 cells
    '''
    mean_number_of_cells = np.mean(yy)

    '''
    We estimate the concentration of cells in each lymph node using the
    mean concentration (units 10^4 cells) and the mean volume of a lymph
    node (estimated from the paper as 400 uL) and we adjust the units
    to obtain 10^3 cells per microliter. The blood data is already in
    concentration units. Hence, we do not scale it.
    '''
    blood_regex = re.compile(r'blood', re.IGNORECASE)
    material_is_not_blood = True
    obj = blood_regex.search(pure_name)

    if obj is not None: 
        material_is_not_blood = False

    if material_is_not_blood: 
        mean_concentration = 10 * mean_number_of_cells /\
                mouse_parameter_table.lymph_node_sample_tissue_volume 
    else:
        mean_concentration = mean_number_of_cells


    '''
    Write to text file 
    '''
    data_to_store = np.concatenate( (parameters, [mean_concentration]) )
    text_file_name = pure_name + '_sinusoid_parameters.txt'

    full_text_file_name = os.path.join(path, text_file_name)

    print full_text_file_name 
    print '-----------------------'

    np.savetxt(full_text_file_name, data_to_store) 

#------------------------------------------------------------
#------------------------------------------------------------
def extract_concentration_data_and_fit_to_sinusoidal(self):
    '''
    This data makes reference to the Circadian paper.
    It describes the concentration of cd8+ cells in different lymphoid
    compartments.
    This function extracts the data points from text files found in a
    given root path and fits them to a sinusoidal curve. 
    The computed parameters ( see fit_data_to_sinusoidal() function ) 
    are stored in a text file.
    '''

    path  = r'/home/rjavier/Documents/pbpk/imm_diag/' +\
            r'point_data_for_examples/'
    regex = re.compile(r'[a-zA-Z0-9_]+[.]csv')

    for dir_path, dir_name, fname in os.walk(path):
        for f in fname: 
            obj = regex.search(f)
            if obj is not None:
                print f
                fit_data_to_sinusoidal(self, dir_path, f)

    print self.mean_concentrations
    print self.labels_for_mean_concentrations

    for value in self.mean_concentrations[1:]:
        print value / self.mean_concentrations[0]

