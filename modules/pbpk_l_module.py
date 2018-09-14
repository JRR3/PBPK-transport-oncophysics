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
np.random.seed(10)
import pandas as pd
import re
import os
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['grid.alpha'] = 0.5
#rc('text', usetex='True')
import matplotlib.pyplot as plt
from compartment_module import Compartment
from species_module import Species


'''
Notation

Lymphatic system
A lymphoid organ as a cluster of lymph nodes and surrounding tissue.
'''

class PBPK_L:
    '''
    This class defines the basic structures of the lymphatic system.
    This class is later incorporated into the PBPK model.
    '''
    def __init__(\
            self,\
            parent\
            ):

        '''
        These variables are defined only for testing purposes
        '''
        self.left_leg_compartment       = 0
        self.right_leg_compartment      = 1
        self.iliac_compartment          = 2
        self.abdominal_compartment      = 3
        self.left_thoracic_compartment  = 4
        self.right_thoracic_compartment = 5
        self.head_compartment           = 6
        self.blood_compartment          = 7
        self.blood_hev                  = 7
        self.blood_cv                   = 8
        self.left_footpad_compartment   = 8 
        self.right_footpad_compartment  = 9

        self.naive_dc_index             = 3
        self.activated_dc_index         = 2
        self.naive_cd8_index            = 1
        self.activated_cd8_index        = 0

        self.generation_rate_index      = 0
        self.death_rate_index           = 1
        self.conversion_rate_index      = 2
        self.constant_rate_index        = 3

        '''
        PBPK_Driver relevant variables
        '''
        ## Reference to the parent class
        self.parent = parent

        ## This is the index of the object inside the PBPK class 
        self.driver_object_index = 0

        ## List of the indices of the compartments inside 
        ## the PBPK_Driver class
        self.driver_list_of_compartment_indices   = []

        ## List of the indices of the tissues inside 
        ## the PBPK_Driver class
        self.driver_list_of_tissue_indices = []

        ## List of the variables inside the PBPK_Driver class
        self.driver_list_of_variables = []
        '''
        END OF PBPK_Driver relevant variables
        '''

        ## Descriptive label for the object. 
        ## Note that we do not allow the user to
        ## choose a label since only one object of 
        ## this class should exist.

        self.label = 'PBPK-L'

        ## Total number of compartments defined in the lymphatic system
        self.n_compartments = 0

        ## Total number of tissues defined inside a Tumor object
        self.n_tissues = 0

        species_source = './csv_files/species_properties/species_labels.csv'
        ## List containing the labels of the species under consideration
        self.species_labels = np.genfromtxt(species_source,\
                dtype='str', delimiter=',')

        ## Number of species per compartment
        self.n_species = len(self.species_labels)

        ## Number of species per compartment
        self.n_variables = 0

        ## Internal time
        self.time   = 0

        ## Total volume of tissue
        self.volume = 0

        ## Total volume of tissue from the previous time step
        self.old_volume = 0

        ## Matrix specifying the connectivity between compartments of this
        ## class.
        self.connectivity_matrix = None

        ## List of the compartment labels.
        self.compartment_labels = []

        ## List of tissue labels.
        self.tissue_labels = []

        ## List of compartments. This list encompases all types of
        ## subcompartments.
        self.compartments = []

        ## Vector of values that define which variables to plot
        self.variables_to_plot = None

        ## List describing the species under consideration. We create a
        ## reference to the list of species stored inside the PBPK class.
        self.species = []

        ## Set the periodicity of the tick marks when plotting data
        self.tick_frequency = 7

        ## Number of local effects to consider within each compartment/layer
        self.n_reaction_parameters = Compartment().n_reaction_parameters 

        ## Initial conditions stored as a vector
        self.initial_conditions_vector = None

        '''
        Mappings
        '''

        '''
        We use the Tissue ---> Compartment map to identify the various blood
        tissues to the blood compartment.
        '''
        self.tissue_index_to_compartment_index = {}

        self.tissue_label_to_index             = {}
        self.tissue_index_to_label             = {}
        self.compartment_label_to_index        = {}
        self.compartment_index_to_label        = {}

        '''
        The variable ---> (species, compartment) map is used in the plotting
        function.
        '''
        self.species_and_compartment_to_variable = {}
        self.variable_to_species_and_compartment = {}
        
        '''
        Define the type of membership for each compartment
        '''
        self.compartment_index_to_membership   = {}

        
        '''
        Plotting variables
        '''
        self.results_figure  = None
        self.results_axes    = None
        #self.species_to_plot ='DC (N)'
        #self.species_to_plot ='CD8+ (N)'

        self.species_to_plot       = None

        self.compartments_to_avoid_plotting = []

        self.compartment_label_to_new_label = None

        self.compartment_label_to_linestyle = None

        self.show_species_in_label = True

        self.list_of_plotting_instructions = []

        '''
        Connection to other compartments: We need to know to which tissues
        from the PBPK class the tumor is connected to. Based on this information
        we can determine which entries of the species matrix have to be
        modified. The connectivity matrix contains this information.
        '''
        ## Indices of the adjacent compartments
        self.adjacent_compartment_indices     = []

        ## Labels of the adjacent compartments
        ## [(Neighbor compartment label, local compartment label)]
        self.adjacent_compartment_label_pairs = []

        ## Indices of the adjacent tissues
        self.adjacent_tissue_indices          = []

        ## List of pairs of indices of the adjacent tissues
        ## [(Neighbor tissue index, local tissue index)]
        self.adjacent_tissue_index_pairs      = []

        ## List of pairs of indices of the adjacent compartments
        ## [(Neighbor compartment index, local compartment index)]
        self.adjacent_compartment_index_pairs = []


#------------------------------------------------------------
#------------------------------------------------------------

        '''IIIIIIIIIIIIII   Initialization   IIIIIIIIIIIIIIII'''

#------------------------------------------------------------
#------------------------------------------------------------


        self.load_connectivity_matrix()
        self.create_compartment_labels()
        self.compute_sizes_of_structures()
        self.create_dictionaries()
        self.load_initial_conditions()
        self.create_compartments()
        self.create_species()
        self.load_species_and_compartment_data()
        self.summary()

#------------------------------------------------------------
#------------------------------------------------------------

        '''MMMMMMMMMMMMMMMM Methods MMMMMMMMMMMMMMMMMMMMMMMMM'''

#------------------------------------------------------------
#------------------------------------------------------------

    def create_compartment_labels(self):
        '''
        Extract compartment labels based on tissue labels
        '''

        regex = re.compile(\
                r'(?P<compartment>[a-zA-Z]+) \((?P<kind>[0-9a-zA-Z_]+)\)')

        last_compartment_label = ''
        for label in self.tissue_labels:
            c_label = label
            obj = regex.search(label)
            if obj is not None:
                '''This compartment is divided into tissues'''
                c_label = obj.group('compartment')
                kind    = obj.group('kind')
                if c_label in self.compartment_labels:
                    continue
            self.compartment_labels.append(c_label)


#------------------------------------------------------------
#------------------------------------------------------------
    def compute_sizes_of_structures(self):

        self.n_compartments    = len(self.compartment_labels)
        self.n_variables       = self.n_compartments * self.n_species
        self.variables_to_plot = np.full(self.n_variables, True)

#------------------------------------------------------------
#------------------------------------------------------------
    def load_connectivity_matrix(self):
        '''
        Generate the connectivity matrix from the diffusive transport matrix.
        The number of tissues is also determined from this data structure.
        The labels are obtained likewise.
        '''

        fname = './csv_files/pbpk_properties/connectivity_matrix.csv'
        self.connectivity_frame = pd.read_csv(fname, index_col=0)
        print self.connectivity_frame
        self.connectivity_frame.fillna(0, inplace=True)

        '''Get labels for tissue: Order sensitive'''
        self.tissue_labels = tuple(self.connectivity_frame.index.values)
        self.n_tissues     = len(self.tissue_labels)

        '''
        The number of tissues was previously found and hence is available for
        the subsequent computations.
        '''
        self.connectivity_matrix = np.zeros((self.n_tissues, self.n_tissues))

        for col_index, col_label in\
                enumerate(self.connectivity_frame.columns.values):
            for row_index, row_label in\
                    enumerate(self.connectivity_frame.index.values):
                value = self.connectivity_frame[col_label][row_label]
                self.connectivity_matrix[row_index, col_index] = value



#------------------------------------------------------------
#------------------------------------------------------------
    def load_initial_conditions(self):
        '''
        Initial conditions for all compartments
        '''

        self.initial_conditions_vector = np.zeros(self.n_variables)

#------------------------------------------------------------
#------------------------------------------------------------
    def create_compartments(self):
        '''
        Create the compartments of the PBPK-L model 
        '''
        for c_index, label in enumerate(self.compartment_labels):
            self.compartments.append(\
                    Compartment(\
                    label,\
                    c_index,\
                    self.compartment_index_to_membership[c_index])\
                    )
                    
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

        '''Map between compartment and membership'''

        regex = re.compile(r'blood', re.IGNORECASE) 
        for k,label in enumerate(self.compartment_labels): 
            obj = regex.search(label)
            if obj is not None: 
                self.compartment_index_to_membership[k] = 'blood'
            else:
                self.compartment_index_to_membership[k] = 'lymphoid_organ'

        '''
        All the tissues have to map to the same defining compartment.
        We use a regular expressions to identify them.
        '''

        txt=r'(?P<compartment>[a-zA-Z_]+([ ][a-zA-Z_]+)?) '+\
                r'\((?P<kind>[0-9a-zA-Z_]+)\)'
        regex = re.compile(txt)

        for k, label in enumerate(self.tissue_labels):
            obj = regex.search(label)
            if obj is not None:
                '''This compartment is divided into tissues'''
                c_label = obj.group('compartment')
                kind    = obj.group('kind')
                index = self.compartment_label_to_index[c_label]
            else:
                index = self.compartment_label_to_index[label]

            self.tissue_index_to_compartment_index[k] = index

        '''
        Create bijection between variable and (species, compartment)
        '''
        for v in range(self.n_variables): 
            compartment, species = np.divmod(v, self.n_species)
            pair = (species, compartment)
            self.variable_to_species_and_compartment[v] = pair
            self.species_and_compartment_to_variable[pair] = v
        
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
    def load_species_and_compartment_data(self):
        '''
        Load species data from text files

        Compartment specific information
        The reaction rates are loaded using files of the form:
        rx_in_ + compartment_label + .txt
        These files have 5 rows by R columns (species x reaction effects)
        Note that the local_reaction_coefficient_matrix is completely
        overwritten.
        '''


        print '*********************Setting reaction coefficient matrix'


        '''Set reaction rates for all species'''
        for k, c in enumerate(self.compartments):

            fname_root = './csv_files/pbpk_properties/reactions/'

            try:
                c_label = c.label
                c_label = c_label.replace(' ', '_')
                c_label = c_label.replace(':', '')
                fname = fname_root + 'rx_in_' + c_label + '.csv'

                frame = pd.read_csv(fname, index_col=0, header=0)
                m = frame.values
                n_rows,n_cols = m.shape


                if np.any(frame.index.values != self.species_labels):
                    print 'Error: Species labels are not compatible'
                    exit()

                if n_rows != self.n_species:
                    print 'Error: # of rows of',
                    print 'local_reaction_coefficient_matrix of',\
                            'compartment', c.label,\
                            'is not equal to n_species'
                    exit()

                if n_cols != self.n_reaction_parameters:
                    print 'Warning: # of columns of',
                    print 'local_reaction_coefficient_matrix of',\
                            'compartment', c.label,\
                            'is not equal to n_reaction_parameters'

                    if n_cols < self.n_reaction_parameters: 
                        print 'The remaining entries will be filled with zeros'
                    else:
                        print 'Error: The data is oversized'
                        exit()

                c.local_reaction_coefficient_matrix[:n_rows, :n_cols] = m


            except:
                print 'File', fname, 'does not exists'
                exit()
                c.local_reaction_coefficient_matrix =\
                        np.zeros((self.n_species,\
                        self.n_reaction_parameters))
                np.savetxt(fname, c.local_reaction_coefficient_matrix,\
                        delimiter = ',')
                print fname, 'was created with zero data'

        '''Set compartment volume'''
        print '*********************Setting compartment volume'

        fname = './csv_files/pbpk_properties/'
        fname += 'compartment_volume.csv'

        try:
            volumes = pd.read_csv(fname, index_col=0, header=0)

        except:
            print 'Error: Unable to read file', fname
            exit()

        for k, c in enumerate(self.compartments):
            c.volume = volumes.loc[c.label].values[0]

        print '*********************Loading transport data for species'

        list_of_matrix_labels      = ['L','A','P','D']
        list_of_species_attributes = \
                ['diffusion_length','contact_area',\
                'partition_coefficient','diffusion_coefficient']

        for index, matrix_label in enumerate(list_of_matrix_labels): 
            attribute = list_of_species_attributes[index] 
            fname_target = './csv_files/pbpk_properties/' + attribute + '/'
            for s in self.species:

                species_label = s.label.replace(' ', '_')
                fname = fname_target + matrix_label +\
                        '_' + species_label + '.csv'
                frame = pd.read_csv(fname, header=0, index_col=0)

                '''
                Check label consistency
                '''
                for i,j in zip(frame.columns.values, self.tissue_labels):
                    if i != j:
                        print 'Column name', i, 'does not match', j
                        exit()

                for i,j in zip(frame.index.values, self.tissue_labels):
                    if i != j:
                        print 'Row name', i, 'does not match', j
                        exit()
                    

                m = frame.values

                if m.shape[0] != m.shape[1]:
                    print 'Error:', fname, 'is not square'
                    exit()

                if matrix_label == 'P':
                    if m.shape[0] != self.n_compartments:
                        print 'Error: Incompatible number of compartments in',\
                                fname
                        exit()
                else:
                    if m.shape[0] != self.n_tissues:
                        print 'Error: Incompatible number of tissues in',\
                                fname
                        exit()

                '''
                Store information in species object
                '''
                setattr(s, attribute, m);


        print '*********************data was loaded successfully'


#------------------------------------------------------------
#------------------------------------------------------------
    def print_connectivity(self):
        '''
        Print connectivity of the elements of this object.
        '''
        for i in range(self.n_tissues): 
            i_label = self.tissue_labels[i]
            for j in range(self.n_tissues): 
                j_label = self.tissue_labels[j]
                if self.connectivity_matrix[i][j] == 1:
                    print '{:<20}'.format(i_label), '--->', j_label


#------------------------------------------------------------
#------------------------------------------------------------
    def print_compartments(self):

        for k, c in enumerate(self.compartments):
           print 'Compartment', k, '--->', c.label 
           print 'Volume:', '{:0.1e}'.format(c.volume), 'mm^3'
           print '------------------'


#------------------------------------------------------------
#------------------------------------------------------------

    def compute_local_reactions(self, dydt, y, t):
        '''
        Compute all contributions due to local compartment effects.
        The vector of concentrations y is assumed constant in
        this function.
        The vector dydt is updated inside this function.
        '''

        '''
        Now we compute the contributions due to reaction terms local to each
        compartment. Hence we call methods of the Compartment class.
        Note that though we pass the full vector of concentrations to the
        compartment, the compartment has already been notified of which
        variables it owns and hence it will only update those.
        '''
        for compartment in self.compartments:
            compartment.time = t
            compartment.update_concentrations(y)
            compartment.compute_local_reaction(dydt) 

#------------------------------------------------------------
#------------------------------------------------------------
    def summary(self):
        '''
        Summarize the properties of the recently created object.
        '''
        print '======================== PBPKL SUMMARY============='
        print 'PPPPPPPP', self.label, 'object created...'
        print '...# of compartments :', self.n_compartments
        print '...# of species      :', self.n_species
        print '...# of Tissues      :', self.n_tissues
        print '...# of variables    :', self.n_variables


        if len(self.species) == 0:
            print 'Species have not been defined yet'

        self.print_compartments()
        self.print_connectivity()

        print '==================END OF PBPKL SUMMARY============='

#------------------------------------------------------------
#------------------------------------------------------------
    def generate_graphical_output(self):
        '''
        Generate graphical output for PBPKL model
        '''

        print self.label, ': Generating graphical output ............' 

        self.results_figure = plt.figure()
        self.results_axes   = self.results_figure.add_subplot(111) 

        for D in self.list_of_plotting_instructions: 

            species_label     = D['species_label']
            species_index     = D['species_index']
            compartment_label = D['compartment_label']
            compartment_index = D['compartment_index'] 
            plot_label        = D['plot_label']
            variable          = D['variable']
            color             = D['color']
            ls                = D['linestyle']
            marker            = D['marker']
            mark_every        = D['mark_every']
            x_data            = D['x_data']
            y_data            = D['y_data']

            self.results_axes.plot(\
                    self.parent.time_vector,\
                    self.parent.solution_vector[:, variable],\
                    color    = color,\
                    label    = plot_label,\
                    marker   = marker,\
                    #markerfacecolor='None',\
                    markevery = mark_every,\
                    linestyle = ls)


        self.results_axes.grid()
        self.results_axes.legend(loc='best')

        self.results_axes.set_xlabel('Time (days)')
        self.results_axes.set_ylabel('Concentration ($10^3$ Cells / $\mu$L)')

        fname = 'results_' + self.label + '.pdf'
        self.results_figure.savefig(fname, format='pdf')

