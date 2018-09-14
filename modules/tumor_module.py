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
import re
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['grid.color'] = 'k'
#mpl.rcParams['grid.alpha'] = 0
#rc('text', usetex='True')
import matplotlib.pyplot as plt
from compartment_module import Compartment
from species_module import Species

#------------------------TUMOR CLASS-------------------------
#------------------------------------------------------------
#------------------------------------------------------------

'''
Notation

Subcompartment
A compartment that lives within another compartment.

Layer
This is a subcompartment of the tumor class and conceptually represents 
a layer of a spheroid tumor. Each tumor is composed of N layers.

HEV
High endothelial venule

CV
Cardiovascular

Tissue
Whenever we have for the same compartment different transport
mechanisms, we distinguish those by means of tissues. For example, the Blood
compartment can be separated into HEV transport and CV transport.
All compartments have at least one type of tissue.

Boundary
This is a compartment associated to the last layer of the tumor.
Hence, layer N can have different transport mechanisms. By default, each
boundary is assumed to have at least one additional transport mechanism. 
Therefore the last layer is composed of more than one type of tissue.

Tumor 1: Layer 0
Tumor 1: Layer 1
...
Tumor 1: Layer N

'''

class Tumor:
    '''
    This class models the behaviour of a tumor within an organism.
    '''
    def __init__(\
            self,\
            parent,\
            label    = 'Tumor',\
            n_layers = 10\
            ):

        # These labels are compatible with the ones defined in the PBPK class
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

        ## Descriptive label for object
        self.label = label

        ## Maximum number of layers for object
        self.n_maximum_layers_for_tumor = n_layers

        ## Number of additional tissues that compose the boundary
        self.n_additional_tissues_for_boundary = 1

        ## Total number of compartments defined inside a Tumor object
        self.n_compartments = self.n_maximum_layers_for_tumor

        ## Total number of tissues defined inside a Tumor object
        self.n_tissues =\
                self.n_maximum_layers_for_tumor +\
                self.n_additional_tissues_for_boundary

        ## Boundary tissue index
        self.boundary_tissue_index = self.n_tissues - 1

        species_source = './csv_files/species_properties/species_labels.csv'
        ## List containing the labels of the species under consideration
        self.species_labels = np.genfromtxt(species_source,\
                dtype='str', delimiter=',')

        ## Number of species per compartment
        self.n_species = len(self.species_labels)

        ## Number of variables
        self.n_variables = self.n_species * self.n_compartments

        ## Internal time
        self.time   = 0

        ## Total volume of tissue
        self.volume = 0

        ## Total area of tissue
        self.area = 0

        ## Radius
        self.radius = 0

        ## Total volume of tissue from the previous time step
        self.old_volume = 0

        ## How many layers of tissue are active or available in the current
        ## state of the model.
        ## For testing purposes we consider all layers to be active.
        self.n_active_compartments = self.n_compartments

        ## Current index of the last active compartment. This should be
        ## n_active_compartments - 1
        self.last_active_layer_index = self.n_active_compartments - 1

        ## Matrix specifying the connectivity between compartments of this
        ## class.
        self.connectivity_matrix =\
                np.zeros((self.n_tissues,self.n_tissues))

        ## Thickness of layers (mm)
        self.layer_thickness = 0.020

        ## Initial radius (mm)
        self.core_radius = 0.100

        ## Vector describing the outer radius of each layer (mm)
        self.outer_radius_vector = np.zeros(self.n_compartments)

        ## Vector describing the inner radius of each layer (mm)
        self.inner_radius_vector = np.zeros(self.n_compartments)

        ## List of the compartment labels.
        self.compartment_labels = []

        ## List of tissue labels.
        self.tissue_labels = []

        ## List of compartments. This list encompases all types of
        ## subcompartments.
        self.compartments = []

        ## Vector of values that define which species to plot in the
        ## generate_graphical_output() function. Its size is equal to
        ## the number of species under consideration.
        self.species_to_plot = np.full(self.n_species, True)

        ## Selected variables to be plotted
        self.variables_to_plot = np.full(self.n_variables, True)

        ## List describing the species under consideration. 
        ## Note that we create a Species object for each species.
        '''
        No reference to parent object.
        '''
        self.species = []

        ## Set the periodicity of the tick marks when plotting data
        self.tick_frequency = 7

        ## Number of local effects to consider within each compartment/layer
        self.n_reaction_parameters = Compartment().n_reaction_parameters

        ## Current volume capacity of the object. This is not the current
        ## volume of the tumor. It is computed by adding the volumes of all the
        ## active compartments. Hence it is larger than or equal to the volume
        ## of the tumor.
        self.current_volume_capacity = 0


        ## Initial conditions stored as a vector
        self.initial_conditions_vector = None

        ## Radial position of the layer taking the center of the tumor as
        ## reference.
        self.layer_center_radial_position_vector =\
                np.zeros(self.n_compartments)

        ## Radial position of the layer taking the center of the tumor as
        ## reference.
        self.layer_vascular_fraction_vector = np.zeros(self.n_compartments)

        ## Radial position of the layer taking the center of the tumor as
        ## reference.
        self.layer_intersection_radial_position_vector =\
                np.zeros(self.n_compartments-1)


        ## Transport coefficient between layers.
        '''
        This vector is expected to be used with the:
        self.layer_intersection_radial_position_vector for plotting purposes.
        Since we have N layers, we have N-1 possible transport connections.
        '''
        self.transport_coefficient_between_layers_vector =\
                np.zeros(self.n_compartments-1)



        ## Volume of each of the compartments that compose this object.
        '''
        All layers are taken into account.
        '''
        self.volume_vector = np.zeros(self.n_compartments)


        ## Diffusion coefficient between layers.
        '''
        This vector is expected to be used for plotting purposes.
        Since we have N layers, we have N-1 possible transport connections.
        Note that theoretically we have n_species options for the diffusion
        coefficient between two layers. However, currently we consider them 
        invariant.
        '''
        self.diffusion_coefficient_between_layers_vector =\
                np.zeros(self.n_compartments-1)

        '''
        Mappings
        '''

        '''
        We use the Tissue ---> Compartment map to identify the boundary
        with the last layer of the tumor.
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


        '''
        Plotting variables
        '''
        self.results_figure = None
        self.results_axes   = None
        self.local_time_horizon = 2
        self.time_vector    = None
        self.solution_vector= None

        '''
        Class specific variables: This variables are unique to the given class
        '''

        '''
        Dimensions  of  Blood  Vessels  from  Distributing
        Artery  to  Collecting  Vein
        By Mary   P.  Wiedeman,  Ph.D.
        Units are in micro meters
        '''
        self.diameter_of_mouse_skeletal_muscle_capillary = 5.6/2 
        self.vessel_radius =\
                self.diameter_of_mouse_skeletal_muscle_capillary / 2
        '''
        Units are in mm
        This data is not used.
        Each compartment has its own definition of vessel radius.
        '''
        self.vessel_radius /= 1000.

        '''
        Lymphathic vessels and high endothelial venules are increased in the
        sentinel lymph nodes of patients with oral squamous cell carcinoma
        beforlymphatice the arrival of tumor cells.
        Surgical oncology
        Units are in micro meter ^ 2
        '''
        self.hpf_area = 193562.0
        self.average_lym_area_in_sentinel_node_positive =\
                2361.8
        self.average_lym_area_in_sentinel_node_control =\
                1621.9
        self.average_hev_area_in_sentinel_node_positive =\
                18258.5
        self.average_hev_area_in_sentinel_node_control =\
                10350.5

        self.average_lym_fraction_positive =\
                self.average_lym_area_in_sentinel_node_positive /\
                self.hpf_area

        self.average_hev_fraction_positive =\
                self.average_hev_area_in_sentinel_node_positive /\
                self.hpf_area

        self.average_lym_fraction_control =\
                self.average_lym_area_in_sentinel_node_control /\
                self.hpf_area

        self.average_hev_fraction_control =\
                self.average_hev_area_in_sentinel_node_control /\
                self.hpf_area

#------------------------------------------------------------
#------------------------------------------------------------

        '''IIIIIIIIIIIIII   Initialization   IIIIIIIIIIIIIIII'''

#------------------------------------------------------------
#------------------------------------------------------------


        self.create_labels()
        self.create_compartments()
        self.create_dictionaries()
        self.load_initial_conditions()
        self.create_species()
        self.load_species_and_compartment_data()
        #self.set_blood_connection()
        #self.set_lymphoid_connection()
        #self.set_sentinel_connection()
        self.summary()


#------------------------------------------------------------
#------------------------------------------------------------

        '''MMMMMMMMMMMMMMMM Methods MMMMMMMMMMMMMMMMMMMMMMMMM'''

#------------------------------------------------------------
#------------------------------------------------------------
    def update_object_area(self):
        '''
        Update the area property of the object.
        This function also returns the area of the object.
        '''

        index = self.last_active_layer_index
        self.area = self.compartments[index].outer_surface_area 
        
        return self.area


#------------------------------------------------------------
#------------------------------------------------------------
    def update_volume_vector(self):
        '''
        Update the vector that contains the volume of each compartment.
        '''

        for index, c in enumerate(self.compartments):
            self.volume_vector[index] = c.volume


#------------------------------------------------------------
#------------------------------------------------------------
    def set_vascular_fraction_of_compartments(self, vec):
        '''
        Warning: It seems that this method is not being used in the class.
        '''

        assert len(vec) == self.n_compartments,\
                'Size of vector incompatible with # of compartments'

        for index, c in enumerate(self.compartments):

            self.check_boundedness_between_0_1(vec[index])
            c.vascular_fraction = vec[index]


#------------------------------------------------------------
#------------------------------------------------------------

    def create_species(self):
        '''
        This function creates a Species object for
        each species listed in the species_labels list.
        Note that initially the species parameter data
        is automatically generated and equal to 1.
        The number of tissues and compartments that define the parameter
        matrices is obtained from the parent creating object.
        '''

        for label in self.species_labels:
            self.species.append( Species(self, label) )


#------------------------------------------------------------
#------------------------------------------------------------

    def compute_and_set_volume_of_layers_from_object_radius(self,\
            print_summary = True):
        '''
        All layers (including the core) get the same thickness.
        '''

        delta_r = self.radius / self.n_compartments
        self.layer_thickness = delta_r
        self.core_radius     = delta_r

        '''
        Geometric porperties are computed
        '''
        self.compute_compartment_properties()

        '''
        Transport properties are computed
        Modifications to parent class take place inside this function.
        '''
        self.update_contact_area_and_diffusion_length()

        if print_summary:
            self.summary()



#------------------------------------------------------------
#------------------------------------------------------------

    def test_1_for_tumor_compartment(self):
        '''
        This function aims to test the connections of the tumor object.
        For this test we have no reaction effects inside the tumor object.
        We connect the blood compartment to the middle layer of the tumor and
        expect to see initially a Gaussian profile.
        '''
        pass



#------------------------------------------------------------
#------------------------------------------------------------
    def load_species_and_compartment_data(self):
        '''
        Load species data from text files

        Compartment specific information
        The reaction rates are loaded using files of the form:
        rx_in_ + compartment_label + .txt
        These files have 5 rows by R columns (species x reaction parameters)
        Note that the local_reaction_coefficient_matrix is completely
        overwritten.
        '''

        print '*********************loading data for compartments'

        print '*********************setting reaction coefficient matrix'
        '''Set reaction rates for all species'''

        for k, c in enumerate(self.compartments):

            try:

                c_label = c.label
                c_label = c_label.replace(' ', '_')
                c_label = c_label.replace(':', '')
                fname = './csv_files/tumor_properties/reactions/'
                fname += 'rx_in_' + c_label + '.csv'

                m = np.genfromtxt(fname, dtype='str', delimiter=',')
                n_rows,n_cols = m.shape

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
                print 'File', fname, 'was found.'

            except:

                print 'File', fname, 'does not exists'
                c.local_reaction_coefficient_matrix =\
                        np.zeros((self.n_species,\
                        self.n_reaction_parameters))
                np.savetxt(fname, c.local_reaction_coefficient_matrix,\
                        delimiter = ',')
                print fname, 'was created with zero data'



#------------------------------------------------------------
#------------------------------------------------------------
    def create_labels(self):

        '''Create labels for the compartments'''

        for c_index in range(self.n_compartments):
            c_label = self.label + ': Layer ' + str(c_index)
            self.compartment_labels.append(c_label)

        '''
        Create the labels of the tumor tissues. All except the last tissue have
        the same label as the corresponding compartment.
        '''

        for t_index in range(self.n_tissues-1):
            t_label = self.label + ': Layer ' + str(t_index)
            self.tissue_labels.append(t_label)


        '''The boundary of the tumor is identified as a separate tissue'''
        t_label = self.label + ' (Boundary)' 
        self.tissue_labels.append(t_label)


#------------------------------------------------------------
#------------------------------------------------------------
    def load_initial_conditions(self):
        '''
        Initial conditions for all compartments
        '''

        self.initial_conditions_vector = np.zeros(self.n_variables)

#------------------------------------------------------------
#------------------------------------------------------------
    def compute_compartment_properties(self):

        '''
        Populate physical properties of each compartment
        '''

        total_volume = 0

        for c_index, c in enumerate(self.compartments):
            '''
            The following function relies on two parameters:
            (0) The core radius
            (1) The CONSTANT thickness of each layer
            '''

            v = self.compute_compartment_geometry_from_index(c_index)

            volume             = v[0]
            inner_surface_area = v[1]
            outer_surface_area = v[2]
            center             = v[3]
            inner_radius       = v[4]
            outer_radius       = v[5]

            c.volume             = volume
            c.inner_surface_area = inner_surface_area 
            c.outer_surface_area = outer_surface_area  
            c.thickness          = self.layer_thickness  

            if c_index < self.n_compartments - 1: 
                self.layer_intersection_radial_position_vector\
                        [c_index]= outer_radius

            '''
            NOTE:
            The center property of the compartment object is with reference to
            all the elements of the PBPK model. Hence this property is a tuple.
            Typically this would either have 2 or 3 elements.
            c.center = center  
            '''

            self.layer_center_radial_position_vector[c_index] = center
            self.inner_radius_vector[c_index] = inner_radius
            self.outer_radius_vector[c_index] = outer_radius

            total_volume += volume

        self.volume = total_volume

#------------------------------------------------------------
#------------------------------------------------------------
    def get_diffusion_length_between_layers(self,\
            layer_1_index, layer_2_index):
        '''
        Get diffusion length between (supposedly) adjacent layers.
        This quantity is computed by taking the average between the 
        thickness of each layer.
        '''
        a = self.compartments[layer_1_index].thickness
        b = self.compartments[layer_2_index].thickness

        return np.mean((a,b))

#------------------------------------------------------------
#------------------------------------------------------------
    def get_intersection_area_between_layers(self,\
            layer_1_index, layer_2_index):
        '''
        Get area of intersection between (supposedly) adjacent layers.
        '''

        a = np.zeros(2)
        b = np.zeros(2)
        tol = 1e-8

        a[0] = self.compartments[layer_1_index].inner_surface_area
        a[1] = self.compartments[layer_1_index].outer_surface_area

        b[0] = self.compartments[layer_2_index].inner_surface_area
        b[1] = self.compartments[layer_2_index].outer_surface_area

        for i in range(2):
            for j in range(2):
                if np.abs(a[i] - b[j]) < tol:
                    return a[i]

        print 'Error: No match between layer contact area'
        exit()

#------------------------------------------------------------
#------------------------------------------------------------
    def set_homogeneous_diffusion_coefficient_between_layers(self, value):
        '''
        This function sets the diffusion coefficient between adjacent layers.
        The parent object gets updated inside this function.
        Note that all layers have the same diffusion coeffcient.
        '''

        '''
        We iterate over all layers and exclude the boundary, which is
        classified as a separate tissue.
        NOTE: In this class tissues and compartments are equivalent.
        '''

        self.diffusion_coefficient_between_layers_vector.fill(value)

        for i in range(self.n_tissues-1):
            for j in range(self.n_tissues-1): 


                '''Check if compartments are adjacent'''
                if self.connectivity_matrix[i][j] == 1:

                    i_label = self.tissue_index_to_label[i]
                    j_label = self.tissue_index_to_label[j]

                    '''
                    Set diffusion coeffient between layers.
                    We update both, the local and global species matrix.
                    '''
                    for s_index, s in enumerate(self.species):

                        '''
                        Local update
                        '''
                        s.symmetric_modification_using_index(\
                                'diffusion_coefficient',\
                                i, j, value)

                        '''
                        Parent update
                        '''
                        self.parent.species[s_index].\
                                symmetric_modification_using_label(\
                                'diffusion_coefficient',\
                                i_label, j_label, value)



#------------------------------------------------------------
#------------------------------------------------------------
    def compute_transport_coefficients_between_adjacent_layers(self):

        '''
        This function sets the diffusion coefficient between adjacent layers.
        '''

        '''
        We iterate over all layers and exclude the boundary, which is
        classified as a separate tissue.
        NOTE: Due to how we iterate over the tissues, the layers should arise
        sequentially increasing by radius. 
        However, it would be desirable to have a more robust
        control on how we determine their sequence.
        '''
        counter = 0

        for i in range(self.n_tissues-1):
            for j in range(self.n_tissues-1): 

                #condition =\
                        #(self.connectivity_matrix[i][j] == 1) or\
                        #(self.connectivity_matrix[j][i] == 1)


                '''Check if compartments are adjacent'''
                if self.connectivity_matrix[i][j] == 1:


                    '''
                    We assume all species share the same transport properties
                    between layers.
                    '''
                    A = self.species[0].contact_area[i,j]
                    L = self.species[0].diffusion_length[i,j]
                    D = self.species[0].diffusion_coefficient[i,j]

                    self.transport_coefficient_between_layers_vector\
                            [counter] = A * D / L 
                    counter += 1



#------------------------------------------------------------
#------------------------------------------------------------

    def update_contact_area_and_diffusion_length(self):
        '''
        This function updates the contact area and diffusion length properties
        between adjacent layers.  
        This properties are derived from geometrical aspects of the layers.
        
        The contact area is simply the area of the
        intersection between the layers. The diffusion length is the average of
        the layer thickness.
        '''

        '''
        We iterate over all layers and exclude the boundary, which is
        classified as a separate tissue.
        
        Note that we also update the properties of the parent object.

        '''

        for i in range(self.n_tissues-1):
            for j in range(self.n_tissues-1): 

                #condition =\
                        #(self.connectivity_matrix[i][j] == 1) or\
                        #(self.connectivity_matrix[j][i] == 1)

                '''Check if compartments are adjacent'''
                if self.connectivity_matrix[i][j] == 1:
                    '''
                    Update diffusion length and contact area between layers.
                    '''
                    diffusion_length =\
                            self.get_diffusion_length_between_layers(i, j)

                    contact_area =\
                            self.get_intersection_area_between_layers(i, j)

                    '''
                    Iterate over all the species.
                    NOTE: The species index is consistent throughout objects.
                    '''
                    for s_index, s in enumerate(self.species):
                        '''
                        Note that by modifying a property of the species
                        object, we have to update the corresponding species
                        object of the parent class.
                        '''

                        '''
                        Changes local to the object
                        '''
                        s.symmetric_modification_using_index(\
                                'contact_area', i, j, contact_area)
                        s.symmetric_modification_using_index(\
                                'diffusion_length', i, j, diffusion_length)



                        '''
                        Changes to the parent object
                        '''
                        i_tissue_label = self.tissue_index_to_label[i]
                        j_tissue_label = self.tissue_index_to_label[j]

                        self.parent.species[s_index].\
                                symmetric_modification_using_label(\
                                'contact_area',\
                                i_tissue_label,\
                                j_tissue_label,\
                                contact_area)

                        self.parent.species[s_index].\
                                symmetric_modification_using_label(\
                                'diffusion_length',\
                                i_tissue_label,\
                                j_tissue_label,\
                                diffusion_length)

                        
                        '''
                        Print data
                        '''
                        if s_index == 0:

                            print i_tissue_label,\
                                    '<--->',\
                                    j_tissue_label,\
                                    'diffusion_length:',\
                                    diffusion_length

                            print i_tissue_label,\
                                    '<--->',\
                                    j_tissue_label,\
                                    'contact_area:',\
                                    contact_area

                            print '-------------------------'



#------------------------------------------------------------
#------------------------------------------------------------
    def create_compartments(self):
        '''
        Create compartments. 
        Additionally the internal connectivity is established. 
        All adjacent compartments are connected.
        '''
        for c_index, c_label in enumerate(self.compartment_labels):
            self.compartments.append(\
                    Compartment(\
                    c_label,\
                    c_index,\
                    'tumor')\
                    )

        self.compute_compartment_properties()

        '''
        Initially we do not consider any kind of local effect.
        The properties of the local_coefficient_matrix are set to zero
        by default. 
        '''
        #c.local_reaction_coefficient_matrix.fill(0.0)


        '''
        Create local connectivity matrix. Note that in this case we have to
        take into account the tissues. The last tissue is assumed to be the
        boundary. It describes a different kind of transport. 
        Hence, whatever is connected to the last active layer of
        the tumor will have to be also connected to the last tissue of the
        tumor.
        '''
        for i in range(self.n_tissues-1):
            for j in range(self.n_tissues-1):
                '''Check if compartments are adjacent'''
                if i + 1 == j: 
                    #print 'Connecting', i, 'with', j
                    self.connectivity_matrix[i][j] = 1


            '''
            The species information is created when the connection to the PBPK
            object is made.
            '''
                    
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

        '''
        Map boundary to last active layer
        '''
        regex = re.compile(r'boundary', re.IGNORECASE)

        for k, label in enumerate(self.tissue_labels):
            obj = regex.search(label)
            if obj is not None:
                index = self.last_active_layer_index
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
    def set_sigmoidal_vascular_fraction(\
            self,\
            max_vascular_fraction = 0.05,\
            min_vascular_fraction = 1e-6,\
            steepness = 6,\
            radius_fraction_for_half_activation = 0.80\
            ):
        '''
        Set a sigmoidal profile for the vascular fraction. The inner part of
        the tumor is assumed to be necrotic.
        '''
        half_radius = self.radius * radius_fraction_for_half_activation

        for k, c in enumerate(self.compartments):

            center = self.layer_center_radial_position_vector[k]
            delta  = center - half_radius 
            argument    = -steepness * delta 
            value  = max_vascular_fraction / ( 1 + np.exp(argument) ) 
            value += min_vascular_fraction

            '''
            Vascular diffusion length and vascular contact area are computed
            inside the call of this function.
            This property is ploted using the plotting_instructions_module
            '''
            c.set_vascular_fraction(value)

            '''
            Keep an internal copy of the values of the vascular fraction.
            '''
            self.layer_vascular_fraction_vector[k] = value


#------------------------------------------------------------
#------------------------------------------------------------
    def set_blood_connection(self):
        '''
        Warning: It seems that this function is not being used.

        When blood perfuses to the tumor one has to take into account the
        vascular fraction within the tissue.

        Diffusion length (ell): It is a function of the vascular fraction.
        ell = 0.5 * R_vessel * (1./ np.sqrt(1./rv) - 1)

        Diffusion coefficient (D): It is set to ten times smaller than the HEV
        diffusion coefficient.

        Contact area (A): This is also a function of the vascular fraction.
        '''

        #(mm)
        manual_diffusion_length = 15. / 1000
        #(mm)^2 / day
        diffusion_coefficient_for_hev = 0.066
        #(mm)^2 / day
        diffusion_coefficient_blood_tumor = diffusion_coefficient_for_hev / 10
        diffusion_length_blood_tumor = diffusion_coefficient_for_hev / 10

        print 'Inside set_blood_connection()'

        for s in self.species: 
            for k, tissue_pair in enumerate(self.adjacent_tissue_index_pairs):
                print 'XXX', self.adjacent_compartment_label_pairs[k]

                neighbor_label, local_label =\
                        self.adjacent_compartment_label_pairs[k] 

                compartment_pair = self.adjacent_compartment_index_pairs[k]


                if neighbor_label != 'Blood':
                    continue

                print neighbor_label, 'and', local_label, 'are connected.'

                tissue_1 = tissue_pair[0]
                tissue_2 = tissue_pair[1]

                compartment_1 = compartment_pair[0]
                compartment_2 = compartment_pair[1]
                
                s.contact_area[tissue_1, tissue_2]           =\
                        1
                s.contact_area[tissue_2, tissue_1]           =\
                        1
                s.diffusion_length[tissue_1, tissue_2]       =\
                        manual_diffusion_length 
                s.diffusion_length[tissue_2, tissue_1]       =\
                        manual_diffusion_length  
                s.diffusion_coefficient[tissue_1, tissue_2]       =\
                        diffusion_coefficient_blood_tumor 
                s.diffusion_coefficient[tissue_2, tissue_1]       =\
                        diffusion_coefficient_blood_tumor 

                s.partition_coefficient[compartment_1, compartment_2]  = 0.2
                s.partition_coefficient[compartment_2, compartment_1]  = 5.0
        

#------------------------------------------------------------
#------------------------------------------------------------
    def activate_sentinel_lymph_node(self):
        pass



#------------------------------------------------------------
#------------------------------------------------------------
    def update_number_of_layers(self):
        '''
        This function checks if the updated volume of the tumor compartment is
        consistent with the number of compartments in the system. If not, we
        can either reduce or increase the number of layers to account for the
        difference.
        '''

        total_volume = 0
        compatible_volume_shell_index = self.n_active_compartments

        for k in range(self.n_active_compartments):
            capacity += self.compartments[k].volume

            if self.volume <= capacity:
                compatible_volume_shell_index = k
                break

        volume_difference = self.volume - self.current_volume_capacity

        if compatible_volume_shell_index == self.last_active_layer_index:
            '''
            We are safe. No need to reduce or increase the number of active
            compartments.
            '''
            pass

        if compatible_volume_shell_index < self.last_active_layer_index:
            '''
            If the volume is reduced and this causes the outermost layer to
            dissapear, then the cells in this layer have to be migrated to the
            next inner layer. This could cause a concentration gradient and the
            system would have to be reequilibrated.
            Note that theoretically it is possible that the volume changed in
            such a significant way that not only the outermost layer dissapears
            but also further inside.
            '''
            '''
            (<) Move cells inside
            '''
            pass

        if self.n_active_compartments == compatible_volume_shell_index:
            '''
            If the volume increases and this causes the object to have a volume
            larger than its current capacity, then new outer layers have to be
            created. This could cause a concentration gradient and the system
            would have to be reequilibrated.
            '''
            n_extra_layers = self.how_many_additional_layers(self,\
                    volume_difference)

#------------------------------------------------------------
#------------------------------------------------------------
    def remove_excedent_layers(self, n_layers): 
        '''
        This function adds the requested number of additional layers that
        have to be activated to accomodate for the volume increase.
        '''

        self.n_active_compartments -= n_layers

        if self.n_active_compartments == 0:
            print 'Tumor has been erradicated'

        if self.n_active_compartments < 0:
            print 'Number of requested layers to be removed ' +\
                    'results in a negative number of compartments'
            exit(0) 

            
#------------------------------------------------------------
#------------------------------------------------------------
    def add_additional_layers(self, n_new_layers): 
        '''
        This function adds the requested number of additional layers that
        have to be activated to accomodate for the volume increase.
        '''

        self.n_active_compartments += n_new_layers
        if self.n_compartments < self.n_active_compartments:
            print 'Number of requested active compartments exceeds ' +\
                    'the number of allocated compartments'
            exit(0) 


#------------------------------------------------------------
#------------------------------------------------------------
    def how_many_additional_layers(self, additional_volume):
        '''
        This function computes the required number of additional layers that
        have to be activated to accomodate for the volume increase.
        '''
        s     = 0
        extra_layer_counter = 0
        while s < additional_volume:
            extra_layer_counter += 1
            index = self.last_active_layer_index + extra_layer_counter
            v = self.compute_compartment_geometry_from_index(index)
            volume = v[0]
            s += volume

        return extra_layer_counter




#------------------------------------------------------------
#------------------------------------------------------------
    def compute_current_volume_capacity(self):
        '''
        Compute the total volume of the object by adding the individual
        contributions of each compartment.
        '''
        s = 0
        for k in range(self.n_active_compartments):
            s += self.compartments[k].volume

        self.current_volume_capacity = s
        return s

#------------------------------------------------------------
#------------------------------------------------------------
    def compute_core_volume_from_radius(self):
        '''
        Standard formula for the volume of a sphere
        '''
        return 4./3. * np.pi * np.power(self.core_radius, 3)

#------------------------------------------------------------
#------------------------------------------------------------
    def compute_core_surface_area_from_radius(self):
        '''
        Standard formula for the surface of a sphere
        '''
        return 4. * np.pi * np.power(self.core_radius, 2)

#------------------------------------------------------------
#------------------------------------------------------------
    def compute_compartment_geometry_from_index(self, index):
        '''
        Return the volume, inner and outer surface area and center
        of the current layer. 
        Note that we assume a spherical geometry. Hence, as the radius 
        increases, each layer of constant thickness has a larger volume and
        surface area.
        Note that we use polar coordinates with respect to the center of the
        tumor.
        '''
        if index == 0:
            inner_radius = 0
            outer_radius = self.core_radius
            volume     =  self.compute_core_volume_from_radius()
            inner_area =  0
            outer_area =  self.compute_core_surface_area_from_radius()
            center     =  0

        else:
            inner_radius = self.core_radius + (index-1) * self.layer_thickness
            outer_radius = inner_radius + self.layer_thickness
            volume =  4./3. * np.pi * ( np.power(outer_radius, 3.) -\
                    np.power(inner_radius, 3.) )
            inner_area =  4. * np.pi * np.power(inner_radius, 2.)
            outer_area =  4. * np.pi * np.power(outer_radius, 2.)
            center     = (outer_radius + inner_radius) / 2.

        return (volume,\
                inner_area, outer_area,\
                center,\
                inner_radius, outer_radius)

           


#------------------------------------------------------------
#------------------------------------------------------------
    def print_compartments(self):

        for k, c in enumerate(self.compartments):
           print 'Compartment', k, '--->', c.label 
           print 'Volume:', '{:0.1e}'.format(c.volume), 'mm^3'
           print '------------------'


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
                    print i_label, '--->', j_label


#------------------------------------------------------------
#------------------------------------------------------------
    def print_relevant_information_between_layers(self):
        '''
        Print the value of transport parameters between connected layers
        '''
        for i in range(self.n_tissues): 
            i_label = self.tissue_labels[i]
            for j in range(self.n_tissues): 
                j_label = self.tissue_labels[j]
                if self.connectivity_matrix[i][j] == 1:
                    print i_label, '--->', j_label
                    for s_index, s in enumerate(self.species):
                        print 'Species:',\
                                self.parent.species_index_to_label[s_index]
                        print '>>>>>>>>>>>>>>>>'
                        print 'Contact area:', s.contact_area[i,j]
                        print 'D. coeff    :', s.diffusion_coefficient[i,j]
                        print 'D. length   :', s.diffusion_length[i,j]
                        print '>>>>>>>>>>>>>>>>'

                    print '****************'


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
        print '======================== TUMOR SUMMARY============='
        print 'TTTTTTTT Tumor object:', self.label, 'created...'
        print '...current volume    :', self.compute_current_volume_capacity()
        print '...# of layers       :', self.n_compartments
        print '...# of active layers:', self.n_active_compartments
        print '...# of species      :', self.n_species
        print '...# of Tissues      :', self.n_tissues
        print '...# of variables    :', self.n_variables
        print 'Current active layer :', self.last_active_layer_index

        for c in self.compartments:
            print 'Tumor subcomparment label  :', c.label
            print 'Tumor subcompartment volume:', c.volume

        print '-------------------'
        print 'Total Tumor volume:', self.volume
        print '-------------------'

        self.print_connectivity()

        if len(self.species) == 0:
            print 'Species have not been defined yet'

        print 'Boundary is linked to tissue:', self.tissue_labels[-1]
        print 'Boundary maps to compartment:',\
                self.tissue_index_to_compartment_index\
                [self.boundary_tissue_index]

        print 'Using HEV fraction (+)   = ',\
                '{:0.2g}'.format(self.average_hev_fraction_positive * 100) +\
                ' %'

        print 'Using LYM fraction (+)   = ',\
                '{:0.2g}'.format(self.average_lym_fraction_positive * 100) +\
                ' %'

        print 'Using HEV fraction (c)   = ',\
                '{:0.2g}'.format(self.average_hev_fraction_control * 100) +\
                ' %'

        print 'Using LYM fraction (c)   = ',\
                '{:0.2g}'.format(self.average_lym_fraction_control * 100) +\
                ' %'

        print '==================END OF TUMOR SUMMARY============='

#------------------------------------------------------------
#------------------------------------------------------------
    def adjust_time_frame(self):

        if  self.local_time_horizon == 0:
            '''
            Clone vectors
            '''
            self.time_vector     = self.parent.time_vector
            self.solution_vector = self.parent.solution_vector
            return

        index = np.argmin(np.abs(\
                self.parent.time_vector - self.local_time_horizon))

        self.time_vector     = self.parent.time_vector[:index]
        self.solution_vector = self.parent.solution_vector[:index,:]

#------------------------------------------------------------
#------------------------------------------------------------
    def extract_all_variables_with_species_index(self, species):

        list_of_variables = []

        for v_index, v in enumerate(self.driver_list_of_variables):

            s_index, c_index =\
                    self.variable_to_species_and_compartment[v_index]

            if s_index != species:
                continue

            list_of_variables.append(v)

        return np.array(list_of_variables, dtype = int)


        
#------------------------------------------------------------
#------------------------------------------------------------
    def generate_snapshot(self, time):
        '''
        Assumptions:

        (0) We plot all the active layers of the tumor.
        (1) Tumor has spherical geometry
        (2) We generate a snapshot for a given instant of time
        '''

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        
        data_index = np.argmin(np.abs(self.parent.time_vector - time))

        s_index = self.parent.species_label_to_index['DC (N)']

        relevant_variables =\
                self.extract_all_variables_with_species_index(s_index)

        time_slot     = self.parent.time_vector[data_index]
        solution_slot = self.parent.solution_vector[data_index,\
                relevant_variables] * 1000

        radii = np.concatenate( ([0], self.outer_radius_vector) )
        solution_slot = np.concatenate( ([solution_slot[0]], solution_slot) )
        azimuth = np.linspace(0, 2*np.pi, 501)

        '''Plot layers of tumor'''
        for k, layer_outer_radius in enumerate(self.outer_radius_vector):
            r_vector = azimuth * 0 + layer_outer_radius
            ax.plot(azimuth, r_vector, 'k-', linewidth = 1)

        r, th = np.meshgrid(radii, azimuth)

        z = np.tile(solution_slot, (r.shape[0], 1))

        cmap = mpl.colors.LinearSegmentedColormap.from_list("", \
                ["blue", "white", "red"])

        '''
        Label editing
        '''
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #angle_in_degrees = 180
        #angle_in_radians = np.deg2rad(angle_in_degrees)
        #ax.set_rlabel_position(angle_in_degrees)
        #shifted_angle    = angle_in_radians + np.deg2rad(2.5)
        #ax.arrow(shifted_angle, 0, 0, 0.25,\
                #head_width=0.05, head_length=0.013, fc='k', ec='k')
        #shifted_angle += np.deg2rad(10)
        #ax.text(shifted_angle, 0.38, 'Radius (mm)')


        '''Plot concentration colormap for layers of tumor'''
        im  = ax.pcolormesh(th, r, z, cmap = cmap)

        #ax.yaxis.grid(color='k', alpha = 1, linewidth=2)

        fig.colorbar(im, label='Concentration (Cells / $\mu$L)')

        txt  = 'Tumor 1: Snapshot at time t = ' +\
                '{:0.2g}'.format(time_slot) + '\n'
        txt += '{:0.2g}'.format(self.n_active_compartments) + ' layers.' +\
                ' Avg. thickness=' +\
                '{:0.2g}'.format(self.layer_thickness) + ' mm'
        fig.suptitle(txt)

        fname = 'radial_' + self.label + '.pdf'
        fig.savefig(fname, format='pdf')


#------------------------------------------------------------
#------------------------------------------------------------
    def set_radii_of_blood_vessels_that_make_up_the_vasculature(self, values):
        '''
        Update the radii of blood vessels of each layer of the tumor.

        These changes take place within the Compartment class.
        '''
        vector = values

        '''
        If input is integer or double then convert to array.
        '''
        if type(values) != np.ndarray:
            vector = values * np.ones(self.n_compartments)

        for k, c in enumerate(self.compartments):
            c.set_blood_vessel_radius(vector[k])

#------------------------------------------------------------
#------------------------------------------------------------
    def set_vascular_fraction(self, values):
        '''
        Update the vascular fraction of a compartment.
        This property is directly related to the connection between a blood
        tissue (Blood (Tumor)) and one or multiple layers.

        Observe that by changing this property, we have to update the contact
        are and diffusion length between the blood tissue and the corresponding
        layers.

        These quantities are computed inside the Compartment class.
        '''
        vector = values

        '''
        If input is integer or double then convert to array.
        '''
        if type(values) != np.ndarray:
            vector = values * np.ones(self.n_compartments)

        for k, c in enumerate(self.compartments):
            '''
            Vascular diffusion length and vascular contact area are computed
            inside the call of this function.
            '''

            c.set_vascular_fraction(vector[k])
            
            '''
            Keep an internal copy of the values of the vascular fraction.
            '''
            self.layer_vascular_fraction_vector[k] = vector[k]




#------------------------------------------------------------
#------------------------------------------------------------
    def generate_graphical_output(self):
        '''
        This function aims to describe the concentration evolution through time
        of the tumor compartment.

        Assumptions:

        (0) We plot all the active layers of the tumor.
        '''

        '''
        Extract a slice of the original time frame for visualization purposes
        '''

        
        self.generate_snapshot(1.0)

        self.adjust_time_frame()

        print self.label, ': Generating graphical output ............'

        colors = ['b','0.45','g','c','y', 'k', 'm', 'r',\
                'maroon', 'gold', 'teal']

        ls = '-'

        self.results_figure = plt.figure()
        self.results_axes   = self.results_figure.add_subplot(111) 

        '''
        Iterate over all variables owned by this object
        '''
        for v_index, v in enumerate(self.driver_list_of_variables):

            if self.variables_to_plot[v_index] == False:
                continue

            s_index, c_index =\
                    self.variable_to_species_and_compartment[v_index]

            '''Only plot naive dendritic cells'''
            '''See lymphoid_species_labels.txt for label names'''
            if self.species_labels[s_index] != 'DC (N)':
                continue

            compartment_label = self.compartments[c_index].label
            species_label     = self.species[s_index].label
            plot_label = compartment_label + ': ' + species_label
            self.results_axes.plot(\
                    self.time_vector,\
                    self.solution_vector[:, v],\
                    color    = colors[c_index],\
                    label    = plot_label,\
                    #marker   = marker_style[s], \
                    #markerfacecolor='None',\
                    #markevery=180,\
                    linestyle=ls)

        self.results_axes.grid()
        self.results_axes.legend(loc='best')

        self.results_axes.set_xlabel('Time (days)')
        self.results_axes.set_ylabel('Concentration ($10^3$ Cells / $\mu$L)')

        fname = 'results_' + self.label + '.pdf'
        self.results_figure.savefig(fname, format='pdf')


