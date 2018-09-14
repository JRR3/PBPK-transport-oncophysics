###############################################
#This project originated at the:
#Source    : Houston Methodist Research Institute
#Location  : Houston, TX
#PI        : Mauro Ferrari
#Supervisor: Arturas Ziemys
#Developer : Javier Ruiz Ramirez
#Origin    : August 2017
###############################################

import time as time_library
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import itertools
mpl.rcParams['lines.linewidth'] = 2
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import mouse_parameter_module
import os
import re
import shutil

import plotting_instructions_module


#------------------------------------------------------------
#------------------------------------------------------------
        
def postprocessing(self):
    '''
    Postprocessing for the Driver class
    '''

    print 'Driver postprocessing ...'

    '''
    Save data to file
    '''
    time_vector_dimension = self.time_vector.shape[0]
    column_labels = ['Time'] + self.variable_labels
    data = np.hstack((self.time_vector.reshape((time_vector_dimension,1)),\
            self.solution_vector))
    frame = pd.DataFrame(data, columns = column_labels)
    fname = './numerical_output/solution.csv'
    frame.to_csv(fname)


    '''
    Compute plotting instructions
    '''
    plotting_instructions_module.compute_plotting_instructions(self);
    print '... Plotting instructions were computed'


    for D_index, D in enumerate(self.plotting_rules_list):

        species_label     = D.species_label
        species_index     = D.species_index
        compartment_label = D.compartment_label
        compartment_index = D.compartment_index 
        plot_label        = D.plot_label
        variable          = D.variable
        color             = D.color
        ls                = D.linestyle
        marker            = D.marker
        mark_every        = D.mark_every
        x_data            = D.x_data
        y_data            = D.y_data
        fun               = D.function
        fig_list          = D.fig_list
        ax_list           = D.ax_list
        peak_data_fname   = D.peak_data_fname
        message           = D.message
        use_separate_figure = D.use_separate_figure
        use_separate_figure = D.use_separate_figure
        plot_activation_function = D.plot_activation_function
        optimize_point_location  = D.optimize_point_location
        x_peak_comparison   = D.x_peak_comparison

        if compartment_label is None:
            compartment_label = ''

        if (x_data is None) and (variable is not None):
            x_data = self.time_vector + 0

        if x_data is not None:
            if self.tick_frequency != 0:
                x_data = x_data / self.tick_frequency


        if (y_data is None) and (variable is not None):
            y_data = fun(self.solution_vector[:, variable])

        '''
        Create as many figures as required
        '''
        for fig_index in fig_list:
            while len(self.list_of_figures) < (fig_index + 1):
                fig = plt.figure()
                fig.add_subplot(111)
                self.list_of_figures.append(fig)


#------------------------------------------------------------
#------------------------------------------------------------
def save_graphical_and_numerical_output_to_file(self, fname):
    '''
    SAVE
    Save data and generate figures.
    '''

    target_path = './graphical_output'
    full_fname = os.path.join(target_path, fname)

    '''
    Extract format identifier
    '''
    regex = re.compile(r'[.](?P<type>\w+)')
    obj = regex.search(fname)
    file_type = ''

    if obj is not None:
        file_type = obj.group('type')
    else:
        print 'No file format identifier found'
        exit()

    if self.use_suptitle_for_plot: 
        self.results_figure.suptitle(txt, fontsize = 10)

    self.results_figure.savefig(full_fname, format = file_type)

