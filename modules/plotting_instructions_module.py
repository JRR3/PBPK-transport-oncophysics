###############################################
#This project originated at the:
#Source    : Houston Methodist Research Institute
#Location  : Houston, TX
#PI        : Mauro Ferrari
#Supervisor: Arturas Ziemys
#Developer : Javier Ruiz Ramirez
#Origin    : August 2017
###############################################

import numpy as np
import pandas as pd
import mouse_parameter_module
from scipy import interpolate
import os
from collections import defaultdict

#------------------------PlottingRules CLASS-----------------
#------------------------------------------------------------
#------------------------------------------------------------
class PlottingRules:

#------------------------------------------------------------
#------------------------------------------------------------
    def __init__(self, D = None):


        self.species_label     = None
        self.species_index     = None
        self.compartment_label = None
        self.compartment_index = None
        self.plot_label        = None
        self.variable          = None
        self.color             = 'b'
        self.linestyle         = '-'
        self.marker            = None
        self.mark_every        = 180
        self.x_data            = None
        self.y_data            = None
        self.function          = lambda x: x
        self.fig_list          = [0]
        self.ax_list           = [0]
        self.peak_data_fname   = None
        self.message           = None
        self.vertical_shift    = None
        self.amplifying_factor = None
        self.horizontal_shift  = None
        self.use_separate_figure = False
        self.plot_activation_function = False
        self.optimize_point_location  = None
        self.file_name         = None
        self.x_label           = None
        self.y_label           = None
        self.use_y_log         = False
        self.x_peak_comparison = None
        self.scale_with_variable = False
        self.scaling_vector    = None
        self.title_information = None

        '''
        If dictionary is passed, use it to populate the class properties.
        '''
        if D is not None:
            self.populate(D)


#------------------------------------------------------------
#------------------------------------------------------------
    def populate(self, D):
        '''
        Populate object through dictionary
        '''
        for key, value in D.iteritems():
            setattr(self, key, value)

#------------------------PlottingRules CLASS-----------------
#-------------------------------END--------------------------
#------------------------------------------------------------



#------------------------------------------------------------
#------------------------------------------------------------

def compute_plotting_instructions(self, additional_data = defaultdict(int)):
    '''
    -------------------------------------------------------
    PPPPPPPPPPPPPPPPPPPPloting
    Plotting instructions for the PBPK_L object
    -------------------------------------------------------
    '''

    '''
    Create PlottingRules objects
    '''
    plotting_rules_map = {}

    #------------------------------------------------------------------D1
    d1 = {}
    compartment_label       = 'Inguinal'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d1['species_label']     = species_label
    d1['species_index']     = species_index
    d1['compartment_label'] = compartment_label
    d1['compartment_index'] = compartment_index
    d1['plot_label']        = compartment_label
    d1['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d1['color']             = 'b'
    d1['linestyle']         = '-'
    d1['marker']            = None
    d1['mark_every']        = 180
    d1['x_data']            = None
    d1['y_data']            = None
    d1['function']          = lambda x: x

    plotting_rules_map[1] = PlottingRules(d1)


    #------------------------------------------------------------------D2
    d2 = {}
    compartment_label       = 'Mesenteric'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d2['species_label']     = species_label
    d2['species_index']     = species_index
    d2['compartment_label'] = compartment_label
    d2['compartment_index'] = compartment_index
    d2['plot_label']        = compartment_label
    d2['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d2['color']             = 'xkcd:tan'
    d2['linestyle']         = '--'
    d2['marker']            = None
    d2['mark_every']        = 180
    d2['x_data']            = None
    d2['y_data']            = None
    d2['function']          = lambda x: x

    plotting_rules_map[2] = PlottingRules(d2)

    

    #------------------------------------------------------------------D3
    d3 = {}
    compartment_label       = 'Right Axillary'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d3['species_label']     = species_label
    d3['species_index']     = species_index
    d3['compartment_label'] = compartment_label
    d3['compartment_index'] = compartment_index
    d3['plot_label']        = compartment_label
    d3['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d3['color']             = 'c'
    d3['linestyle']         = '-'
    d3['marker']            = None
    d3['mark_every']        = 180
    d3['x_data']            = None
    d3['y_data']            = None
    d3['function']          = lambda x: x

    plotting_rules_map[3] = PlottingRules(d3)

    #------------------------------------------------------------------D4
    d4 = {}
    compartment_label       = 'Cervical'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d4['species_label']     = species_label
    d4['species_index']     = species_index
    d4['compartment_label'] = compartment_label
    d4['compartment_index'] = compartment_index
    d4['plot_label']        = compartment_label
    d4['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d4['color']             = 'maroon'
    d4['linestyle']         = '-'
    d4['marker']            = None
    d4['mark_every']        = 180
    d4['x_data']            = None
    d4['y_data']            = None
    d4['function']          = lambda x: x

    plotting_rules_map[4] = PlottingRules(d4)

    #------------------------------------------------------------------D5
    d5 = {}
    compartment_label       = 'Blood'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d5['species_label']     = species_label
    d5['species_index']     = species_index
    d5['compartment_label'] = compartment_label
    d5['compartment_index'] = compartment_index
    d5['plot_label']        = compartment_label
    d5['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d5['color']             = 'xkcd:orange'
    d5['linestyle']         = '-'
    d5['marker']            = None
    d5['mark_every']        = 180
    d5['x_data']            = None
    d5['y_data']            = None
    d5['function']          = lambda x: x


    plotting_rules_map[5] = PlottingRules(d5)


    #------------------------------------------------------------------D6
    d6 = {}
    compartment_label       = 'Spleen'
    species_label           = 'CD8+ N'
    species_index           = self.species_label_to_index[species_label]
    compartment_index       = self.compartment_label_to_index\
            [compartment_label]
    d6['species_label']     = species_label
    d6['species_index']     = species_index
    d6['compartment_label'] = compartment_label
    d6['compartment_index'] = compartment_index
    d6['plot_label']        = 'Spleen'
    #d6['plot_label']        = 'Naive CD8+'
    d6['variable']          =\
            self.species_and_compartment_to_variable[\
            (species_index, compartment_index)]
    d6['color']             = 'orange'
    d6['linestyle']         = '-'
    d6['marker']            = None
    d6['mark_every']        = 180
    d6['x_data']            = None
    d6['y_data']            = None
    d6['function']          = lambda x: x

    plotting_rules_map[6] = PlottingRules(d6)


    #------------------------------------------------------------------STORAGE
    '''
    Store dictionaries inside list of plotting instructions
    (1) Inguinal (Iliac)      CD8+ (N)
    (2) Mesentery (Abdominal) CD8+ (N)
    (3) Axillary (R Thor)     CD8+ (N)
    (4) Cervical (Head)       CD8+ (N)
    (5) Blood                 CD8+ (N)
    (6) Spleen                CD8+ (N)
    '''

    plot_list = [1,2,3,4,5,6]

    self.plotting_rules_list = [plotting_rules_map[x] for x in plot_list] 










