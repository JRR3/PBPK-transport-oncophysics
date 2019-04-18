###############################################
#This project originated at the:
#Source    : Houston Methodist Research Institute
#Location  : Houston, TX
#PI        : Mauro Ferrari
#Supervisor: Arturas Ziemys
#Developer : Javier Ruiz Ramirez
#Origin    : August 2017
#Modified  : April 14, 2019
###############################################

#------------------------------------LIBRARIES
import numpy as np
import re
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import time as time_library
import os
import sys

#------------------------------------Plotting style
mpl.rcParams['lines.linewidth'] = 2

#------------------------------------Module folder
sys.path.insert(0, './modules')

from compartment_module import Compartment
from species_module import Species
from pbpk_l_module import PBPK_L
import postprocessing_module
import mouse_parameter_module


class PBPK_Driver:
    '''
    This class puts together all the components of the PBPK-Lymphatic model
    '''
    def __init__(self):

        print 'IIIIIIIIIIIIIIIII DRIVER INITIALIZATION IIIIIIIIIIII'
        print 'I exist'

        ''' Properties'''
        
        self.label = 'Driver'

        ## This list stores all the objects that own 
        ## compartments in the PBPK_Driver class.
        self.list_of_objects = []

        self.compartment_labels = []

        self.tissue_labels   = []

        self.compartments    = []

        self.species         = []

        self.variable_labels = []

        species_source = './csv_files/species_properties/species_labels.csv'
        ## List containing the labels of the species under consideration
        self.species_labels = np.genfromtxt(species_source,\
                dtype='str', delimiter=',')

        ## Number of species per compartment
        self.n_species = len(self.species_labels)

        self.n_compartments  = 0

        self.n_tissues       = 0

        self.n_variables     = 0

        self.n_tumors        = 0

        self.n_sentinels     = 0

        self.connectivity_matrix = None

        ## We use this map in the compute_diffusive_transport() function.
        ## Since it is natural to iterate over the species and compartments,
        ## it becomes necessary to identify to which variable they are related.
        self.species_and_compartment_to_variable = {}

        ## We use this map in the compute_diffusive_transport() function.
        ## Since it is natural to iterate over the species and tissues,
        ## it becomes necessary to identify to which compartment a tissue
        ## belongs to.
        self.tissue_index_to_compartment_index = {}


        '''
        Lists of objects
        '''
        self.pbpk              = None
        self.tumors            = []
        self.sentinels         = []
        self.additional_organs = []
        self.injections        = []

        '''
        Mappings
        '''
        self.tissue_label_to_index      = {}
        self.tissue_index_to_label      = {}
        self.compartment_label_to_index = {}
        self.compartment_index_to_label = {}
        self.species_label_to_index     = {}
        self.species_index_to_label     = {}

        '''
        Types of tissue: We classify each tissue in 4 different types:

        (0) Blood
        (1) Lymphoid compartment (Includes sentinel lymph nodes)
        (2) Tumor
        (3) Boundary (Includes tumor and sentinel lymph node)

        We use this classification to determine the type of transport that
        takes place between tissues. 

        ??? If we have two tissue, at least one with boundary and separated,
        what should we return for the diffusion length between their
        boundaries?
        Should we ignore this computation whenever the objects are separated?
        '''
        ## Mapping from the tissue type label to the tissue type index
        self.tissue_type_label_to_tissue_type_index =\
                {'blood': 0, 'lymphoid': 1, 'tumor': 2, 'boundary': 3}

        ## We use this dictionary to map the result of the function
        ## (x,y) ---> x + y^2 to consecutive integers.
        self.tissue_pair_index_to_consecutive_index =\
                {0:0, 2:1, 6:2, 12:3, 1:4, 4:5, 9:6, 5:7, 10:8, 11:9}

        ## Mapping from the tissue type label to the tissue type index
        self.tissue_index_to_tissue_type_label= {}

        ## Mapping from the tissue type label to the tissue type index
        self.tissue_index_to_tissue_type_index= {}



        '''
        Plotting variables
        '''
        self.list_of_plotting_instructions = []
        self.results_figure = []
        self.use_suptitle_for_plot = False
        self.tick_frequency = 0


        self.list_of_figures = []

        '''
        SHORT COMPUTATION
        '''
        self.use_short_computation = False
        self.use_short_computation = True

        if self.use_short_computation: 
            self.final_time = 10
        else:
            self.final_time = 50

        '''
        Solver variables
        '''
        #TODO: This should be defined by the user.
        self.dt = 0.2

        '''
        NOTE: Before commiting a result to the log file, make sure that the
        results are consistent using a finer time step.
        '''
        self.n_time_steps  = (self.final_time) / self.dt
        self.time_vector   = np.linspace(0,self.final_time, self.n_time_steps)

        self.initial_conditions_vector = None
        self.solution_vector           = None
        self.time_to_solve_system      = np.inf

        '''
        USE LOG SCALE
        '''
        self.use_logarithmic_scale = False

        self.show_dc_concentrations = True
        self.show_ratio_function    = False

        '''
        Optimization variables
        '''
        self.objective_function_counter = 0
        self.optimization_parameters    = None
        self.optimization_parameters_dictionary = None
        self.index_string_for_new_file  = None
        self.current_call_is_a_repetition = False
        self.parameter_change_vector    = None 

        '''
        Plotting variables
        '''
        self.plotting_rules_list = None




        '''MMMMMMMMMMMMMMMM Methods MMMMMMMMMMMMMMMMMMMMMMMMM'''

        self.create_basic_objects()
        self.populate_list_of_objects()
        self.create_and_assign_variables()
        self.load_initial_conditions()
        self.construct_connectivity()
        self.create_dictionaries()
        self.create_species()
        self.load_species_data()
        self.summary()
        self.solve_ode_system()
        self.postprocessing()
        self.generate_local_graphical_output()

#------------------------------------------------------------
#------------------------------------------------------------
    def load_initial_conditions(self):
        '''
        Load the initial conditions (concentration) of each species.
        '''
        src = './csv_files/pbpk_properties/initial_conditions.csv'
        frame = pd.read_csv(src, index_col=0, header=0)
        frame.fillna(0,inplace=True)


        self.initial_conditions_vector = np.zeros(self.n_variables)

        for c_index, c_label in enumerate(self.compartment_labels):
            for s_index, s_label in enumerate(self.species_labels):
                pair = (s_index, c_index)
                variable = self.species_and_compartment_to_variable[pair]
                value = frame.loc[c_label, s_label]
                self.initial_conditions_vector[variable] = value


#------------------------------------------------------------
#------------------------------------------------------------
    def postprocessing(self):

        '''
        Postprocessing routine generates additional output and formats the data
        for plotting.
        ''' 

        postprocessing_module.postprocessing(self)

#------------------------------------------------------------
#------------------------------------------------------------
    def map_tissue_type_pair_to_index(self,\
            tissue_1_type_index,\
            tissue_2_type_index): 
        '''
        This function takes two tissue type indices and converts them to a
        single index in a injective fashion. This allows us to identify the 
        type of connection of tissues we are looking at. The initial 
        mapping is:
        (x,y) ---> x + y^2 for x <= y and x,y integers.
        Then we map the result using the function
        tissue_pair_index_to_consecutive_index() to obtain:

        {
            (0, 0)
            (2, 1)
            (6, 2)
            (12,3)
            (1, 4)
            (4, 5)
            (9, 6)
            (5, 7) 
            (10,8) 
            (11,9) 
        }
        '''

        a = tissue_1_type_index
        b = tissue_2_type_index

        '''Guarantee that a <= b'''
        if a < b:
            a,b = b,a

        index = int(a + b**2)

        return self.tissue_pair_index_to_consecutive_index[index]


#------------------------------------------------------------
#------------------------------------------------------------
    def create_basic_objects(self):

        print '++++++++++++Creating basic objects+++++++++++++++'

        self.pbpk = PBPK_L(self)


#------------------------------------------------------------
#------------------------------------------------------------
    def populate_list_of_objects(self):
        '''
        Create list of objects whose defining class contains compartments.
        Hence, an injection would not be found in this list.
        '''

        print '++++++++++++Populating list of objects+++++++++++'

        self.list_of_objects.append(self.pbpk)

        self.n_tumors = len(self.tumors)

        for t in self.tumors: 
            self.list_of_objects.append(t)


#------------------------------------------------------------
#------------------------------------------------------------
    def create_and_assign_variables(self):
        '''
        Iterate over all objects and create/update variables based on the
        number of compartments/tissues that it owns.
        Question: Do we need a map from variable to species and compartment?
        '''

        '''
        The variable counter is incremented by the number of variables that
        every object defines. Then they are distributed among the compartments
        owned by that object.
        '''

        print '++++++++++++Assigning variables +++++++++++++++++'

        variable_counter    = 0
        compartment_counter = 0
        tissue_counter      = 0
        #list_of_initial_conditions = []

        for obj_index, obj in enumerate(self.list_of_objects):

            n_tissues_in_object      = obj.n_tissues
            n_compartments_in_object = obj.n_compartments
            n_variables_in_object    = obj.n_variables

            compartment_indices =\
                    range(compartment_counter, compartment_counter +\
                    n_compartments_in_object)

            tissue_indices = range(tissue_counter, tissue_counter +\
                    n_tissues_in_object)

            variable_indices = range(variable_counter, variable_counter +\
                    n_variables_in_object)

            '''Get list of compartments'''
            self.compartment_labels += obj.compartment_labels

            '''Get list of labels'''
            self.tissue_labels += obj.tissue_labels

            '''Set properties of the object'''
            obj.driver_object_index                = obj_index
            obj.driver_list_of_compartment_indices = compartment_indices
            obj.driver_list_of_tissue_indices      = tissue_indices
            obj.driver_list_of_variables           = variable_indices


            '''Get initial conditions'''
            #list_of_initial_conditions.append(obj.initial_conditions_vector)

            for c in obj.compartments:
                '''
                Store a reference to the compartment. 
                '''
                self.compartments.append(c)

                '''
                We need to update the variable indices of each compartment
                '''
                variables = range(variable_counter, variable_counter +\
                        self.n_species)

                ''' Set properties of compartment '''
                c.driver_compartment_index = compartment_counter
                c.driver_variables         = variables

                for s_index in range(self.n_species):

                    pair = (s_index, compartment_counter)

                    ''' Update (species, compartment)--->variable map'''
                    self.species_and_compartment_to_variable[pair] =\
                            variable_counter

                    ''' Update variable counter'''
                    variable_counter    += 1

                ''' Update compartment counter'''
                compartment_counter += 1

            for t_index in range(n_tissues_in_object): 

                #'''Update tissue ---> object map''' 
                #self.tissue_index_to_object[tissue_counter] = obj
                #print 'T_index:', t_index

                '''Compartment index of tissue as seen from the object'''
                local_c_index =\
                        obj.tissue_index_to_compartment_index[t_index]
                driver_c_index =\
                        obj.compartments[local_c_index].\
                        driver_compartment_index 

                '''Update tissue ---> compartment map''' 
                self.tissue_index_to_compartment_index[tissue_counter] =\
                        driver_c_index

                '''Update tissue counter''' 
                tissue_counter += 1


        self.n_variables    = variable_counter
        self.n_compartments = compartment_counter
        self.n_tissues      = tissue_counter
        #self.initial_conditions_vector =\
                #np.concatenate(list_of_initial_conditions)

#------------------------------------------------------------
#------------------------------------------------------------

    def ode_system(self, y, t):
        '''
        This function is the typical function that is passed as an argument
        to an ODE solver. Hence, this function receives the concentrations
        (y) at the current time (t) and returns the rates of change for
        each variable.
        '''
        dydt = np.zeros(self.n_variables)

        '''
        Transport
        '''
        self.compute_diffusive_transport(dydt, y, t)

        '''
        Rx
        '''
        for compartment in self.compartments:
            compartment.time = t
            compartment.update_concentrations(y)
            compartment.compute_local_reaction(dydt) 

        return dydt


#------------------------------------------------------------
#------------------------------------------------------------

    def create_dictionaries(self):

        '''Bijective map between tissue and index'''

        for k,label in enumerate(self.tissue_labels):
            self.tissue_label_to_index[label]       = k
            self.tissue_index_to_label[k    ]       = label 

        '''Bijective map between compartment and index'''

        for k,label in enumerate(self.compartment_labels):
            self.compartment_label_to_index[label]       = k 
            self.compartment_index_to_label[k    ]       = label 

        '''Bijective map between species and index'''
        for k,label in enumerate(self.species_labels):
            self.species_label_to_index[label]      = k
            self.species_index_to_label[k    ]      = label 

        '''
        Create a map from tissue index to tissue type

        (0) Blood
        (1) Lymphoid compartment (Includes sentinel lymph nodes)
        (2) Tumor
        (3) Boundary (Includes tumor and sentinel lymph node)
        '''
        boundary_regex  = re.compile(r'boundary', re.IGNORECASE)
        tumor_regex     = re.compile(r'tumor',    re.IGNORECASE)
        #sentinel_regex  = re.compile(r'sentinel', re.IGNORECASE)
        blood_regex     = re.compile(r'blood',    re.IGNORECASE)
        blood_hev_regex = re.compile(r'hev',      re.IGNORECASE)

        '''Order is relevant'''
        regex_list = [\
                blood_regex,\
                boundary_regex,\
                tumor_regex,\
                #sentinel_regex,\
                ]

        '''
        Print mapping from tissue to type
        '''
        print '>>>>>>>>>>>>>>>>>>>>>>Mapping tissue to type...'
        for t_index,label in enumerate(self.tissue_labels):
            found_it = False
            for r_index, regex in enumerate(regex_list):
                obj = regex.search(label)
                if obj is not None:
                    tissue_type_label = regex.pattern
                    found_it = True
                    break
            if found_it == False: 
                '''
                If not found then it must be a lymphoid organ.
                '''
                tissue_type_label = 'lymphoid'
            print '{:30.30}'.format(label), '--->', tissue_type_label
            self.tissue_index_to_tissue_type_label[t_index] = tissue_type_label

            tissue_type_index =\
                    self.\
                    tissue_type_label_to_tissue_type_index[tissue_type_label]

            self.tissue_index_to_tissue_type_index[t_index] =\
                    tissue_type_index



        '''
        Create labels for variables
        Recall that the species objects have not yet been created
        '''
        self.variable_labels = [''] * self.n_variables

        for s_index, s_label in enumerate(self.species_labels):
            for c_index, c in enumerate(self.compartments):
                pair = (s_index, c_index)
                variable = self.species_and_compartment_to_variable[pair]
                label = c.label + '-->' + s_label
                self.variable_labels[variable] = label



#------------------------------------------------------------
#------------------------------------------------------------

    def compute_diffusive_transport(self,\
            output,\
            concentration_vector,\
            time):
        '''
        Compute all the diffusion fluxes between adjacent compartments. 
        It is essential to know the connection between all compartments from
        all objects to correctly account for all fluxes. Note also that it is
        necessary to know to which compartment a tissue belongs to.
        '''

        for s in range(self.n_species):
            for i in range(self.n_tissues): 
                i_c = self.tissue_index_to_compartment_index[i]
                for j in range(self.n_tissues): 
                    j_c = self.tissue_index_to_compartment_index[j]

                    '''
                    NOTE: The non-symmetry of the connectivity matrix is 
                    relevant.
                    Hence we can not use the function 
                    tissues_are_not_connected(i,j)
                    '''
                    if self.connectivity_matrix[i,j] == 0: 
                        continue

                    v_i = self.species_and_compartment_to_variable[(s, i_c)]
                    v_j = self.species_and_compartment_to_variable[(s, j_c)]
                    i_concentration = concentration_vector[v_i]
                    j_concentration = concentration_vector[v_j]
                    diffusion  = self.species[s].compute_diffusion(i, j,\
                            i_concentration, j_concentration)
                    output[v_j] += diffusion / self.compartments[j_c].volume
                    output[v_i] -= diffusion / self.compartments[i_c].volume

                    #'''Write to text file'''
                    #self.update_flux_file(time, i_c, j_c, diffusion,\
                            #i_concentration, j_concentration)


#------------------------------------------------------------
#------------------------------------------------------------
    def construct_connectivity(self):

        self.connectivity_matrix =\
                np.zeros((self.n_tissues, self.n_tissues), dtype = int)

        print '++++++++++++Building connectivity +++++++++++++++'

        tissue_counter      = 0

        for obj_index, obj in enumerate(self.list_of_objects):

            '''Set properties of the object'''

            n_tissues_in_object = obj.n_tissues

            tissue_indices = range(tissue_counter, tissue_counter +\
                    n_tissues_in_object)

            start  = tissue_counter
            finish = tissue_counter + n_tissues_in_object

            self.connectivity_matrix[start:finish, start:finish] =\
                    obj.connectivity_matrix

            '''Update outer counters'''
            tissue_counter += n_tissues_in_object
    
        self.print_connectivity()


#------------------------------------------------------------
#------------------------------------------------------------
    def print_connectivity(self):
        '''
        Print connectivity of the elements of this object.
        '''

        print '++++++++++++Printing connectivity +++++++++++++++'

        for i in range(self.n_tissues): 
            i_label = self.tissue_labels[i]
            for j in range(self.n_tissues): 
                j_label = self.tissue_labels[j]
                if self.connectivity_matrix[i][j] == 1:
                    print '{:<20}'.format(i_label), '--->', j_label



#------------------------------------------------------------
#------------------------------------------------------------
    def solve_ode_system(self):
        '''
        Solve the system of ODE's using LSODE Fortran solver.
        '''

        if self.current_call_is_a_repetition:
            return

        print '....................Solving the ODE system'

        t1 = time_library.time()

        self.solution_vector =\
                odeint(\
                self.ode_system,\
                self.initial_conditions_vector, \
                self.time_vector,\
                printmessg = True)

        t2 = time_library.time()

        t_elapsed = t2 - t1


        self.time_to_solve_system = t_elapsed


#------------------------------------------------------------
#------------------------------------------------------------
    def print_tissue_to_compartment(self):
        for key, value in\
                self.tissue_index_to_compartment_index.\
                iteritems():
            tissue_label      = self.tissue_labels[key]
            compartment_label = self.compartment_labels[value]
            print tissue_label, '--->', compartment_label

#------------------------------------------------------------
#------------------------------------------------------------

    def create_species(self):
        '''
        This function creates a Species object for
        each species listed in the species_labels list.
        Note that initially the species parameter data
        is automatically generated and equal to 1.
        '''

        for label in self.species_labels:
            self.species.append( Species(self, label) )


#------------------------------------------------------------
#------------------------------------------------------------

    def load_species_data(self):
        '''
        Load species data from each of the objects.  Recall that we are
        basically creating a block diagonal matrix with the diagonal being
        populated with the species properties between each pair of tissues
        defined inside a given object.
        '''

        t_counter = 0
        c_counter = 0
        for obj_index, obj in enumerate(self.list_of_objects): 

            t_dim = obj.n_tissues
            c_dim = obj.n_compartments
            t1 = t_counter
            t2 = t_counter + t_dim
            c1 = c_counter
            c2 = c_counter + c_dim

            t_counter += t_dim
            c_counter += c_dim

            for s_index, s in enumerate(self.species):

                s.contact_area[t1:t2, t1:t2] =\
                        obj.species[s_index].contact_area

                s.diffusion_length[t1:t2, t1:t2] =\
                        obj.species[s_index].diffusion_length

                s.diffusion_coefficient[t1:t2, t1:t2] =\
                        obj.species[s_index].diffusion_coefficient

                s.partition_coefficient[c1:c2, c1:c2] =\
                        obj.species[s_index].partition_coefficient


#------------------------------------------------------------
#------------------------------------------------------------

    def summary(self):
        '''
        Summarize the properties of the recently created object.
        '''

        print '======================== DRIVER SUMMARY============='
        print 'DDDDDDDD', self.label, 'object created...'
        print '...# of species      :', self.n_species
        print '...# of compartments :', self.n_compartments
        print '...# of Tissues      :', self.n_tissues
        print '...# of variables    :', self.n_variables
        print '...# of Tumors       :', self.n_tumors


        if len(self.species) == 0:
            print 'Species have not been defined yet'

        #self.print_connectivity()
        #self.print_tissue_to_compartment()

        print '==================END OF DRIVER SUMMARY============='

#------------------------------------------------------------
#------------------------------------------------------------
    def generate_graphical_output(self):
        '''
        Generate graphical output for all objects
        '''

        t1 = time_library.time()

        for obj in self.list_of_objects: 
            print 'Driver: Generating graphical output ............'
            obj.generate_graphical_output()

        t2 = time_library.time()

        t_elapsed = t2 - t1

        txt = '{:0.2g}'.format(t_elapsed)
        print txt, 'seconds to generate graphical output'



#------------------------------------------------------------
#------------------------------------------------------------
    def generate_local_graphical_output(self):
        '''
        Generate graphical output for selected models
        '''
        print self.label, ': Generating graphical output ............' 

        for D in self.plotting_rules_list: 

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
            vertical_shift    = D.vertical_shift
            amplifying_factor = D.amplifying_factor 
            plot_activation_function = D.plot_activation_function
            x_label           = D.x_label
            y_label           = D.y_label
            use_y_log         = D.use_y_log
            scale_with_variable = D.scale_with_variable
            title_information = D.title_information 

            
            #print 'Plot label:', plot_label

            if scale_with_variable:
                y_exp_max = np.max(y_data)
                y_num_max = np.max(fun(self.solution_vector[:, variable]))
                y_data *= y_num_max / y_exp_max


            if (x_data is None) and (variable is not None):
                x_data = self.time_vector + 0

            if x_data is not None:
                if self.tick_frequency != 0:
                    x_data = x_data / self.tick_frequency

            if (y_data is None) and (variable is not None):
                y_data = fun(self.solution_vector[:, variable])



            for fig_index in fig_list:


                fig = self.list_of_figures[fig_index]
                fig_axes = fig.axes


                if (vertical_shift is not None) or\
                        (amplifying_factor is not None): 
                    original_y_data = y_data + 0

                if (vertical_shift is not None) and\
                        (vertical_shift[fig_index] != 0):
                        y_data += vertical_shift[fig_index]

                if (amplifying_factor is not None) and\
                        (amplifying_factor[fig_index] != 0):
                        y_data *= amplifying_factor[fig_index]

                for ax_index in ax_list:

                    ax = fig_axes[ax_index]

                    if x_label is not None:
                        ax.set_xlabel(x_label)

                    if y_label is not None:
                        ax.set_ylabel(y_label)


                    if use_y_log:

                        ax.semilogy(\
                                x_data,\
                                y_data,\
                                color    = color,\
                                label    = plot_label,\
                                marker   = marker,\
                                #markerfacecolor='None',\
                                markevery = mark_every,\
                                linestyle = ls)

                    else:
                        ax.plot(\
                                x_data,\
                                y_data,\
                                color    = color,\
                                label    = plot_label,\
                                marker   = marker,\
                                #markerfacecolor='None',\
                                markevery = mark_every,\
                                linestyle = ls)


                if (vertical_shift is not None) or\
                        (amplifying_factor is not None):
                    y_data = original_y_data + 0



        '''
        File identifier
        '''
        for fig_index, fig in enumerate(self.list_of_figures):
            axes = fig.axes
            ax = axes[0]
            ax.grid()
            leg = ax.legend(loc=1)

            if fig_index < 1:

                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Concentration ($10^3$ Cells / $\mu$L)')

                if self.tick_frequency != 0:
                    ax.set_xlabel('Time (weeks)')

            if 0 < fig_index:
                fname = 'results_' + self.label + '_' + str(fig_index) + '.pdf'
                fig.savefig(fname)


            fname = 'results_' +\
                    self.label +\
                    '_' +\
                    str(fig_index) +\
                    '.pdf'
            path   = r'./graphical_output'
            fname = os.path.join(path, fname)
            fig.savefig(fname)



        #fname = 'results_' + self.label + '.pdf'
        #self.results_figure = self.list_of_figures[0]
        #postprocessing_module.\
                #save_graphical_and_numerical_output_to_file(self, fname)







#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#-----------Normal execution---------------------------------                


#s      = geometry_module.Sphere()
driver = PBPK_Driver()





