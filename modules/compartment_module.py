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
#from matplotlib import rc
#rc('text', usetex='True')
import matplotlib.pyplot as plt

#------------------------------------------------------------
#------------------------COMPARTMENT CLASS-------------------
#------------------------------------------------------------
#------------------------------------------------------------

class Compartment:
    '''
    This class aims to describe a generic compartment
    '''
    def __init__(self,\
            label               = '',\
            compartment_index   = 0,\
            membership          = 'undefined'
            ):

        '''
        We do not store a pointer to the parent object in view that we do not
        need any specific functionality from that object.
        '''

        ## Compartment name
        self.label = label

        ## Compartment index. 
        ## This is the numbering assigned by the parent object.
        self.compartment_index = compartment_index

        '''
        PBPK_Driver relevant variables
        '''
        ## PBPK compartment index. 
        ## This is the compartment numbering assigned by the driver.
        self.driver_compartment_index = 0

        ## Recall that within each compartment we have a certain number of
        ## species and each pair (species, compartment) corresponds to a
        ## variable. Hence, we store the indices given by PBPK_Driver class.
        self.driver_variables = []
        '''
        END OF PBPK_Driver relevant variables
        '''


        ## Define type of compartment. Currently we have 4 types of
        ## membership.
        ## (0) Lymphoid organ
        ## (1) Blood
        ## (2) Tumor
        ## (3) Sentinel lymph node
        self.membership = membership

        species_source = './csv_files/species_properties/species_labels.csv'
        ## List containing the labels of the species under consideration
        self.species_labels = np.genfromtxt(species_source,\
                dtype='str', delimiter=',')

        ## Map species label to index
        self.species_label_to_index = {}

        ## Number of species per compartment
        self.n_species = len(self.species_labels)


        ## Volume of compartment
        self.volume = 0

        ## Inner surface area of the compartment. This property is only
        ## relevant when the compartment has a well-defined inner surface.
        ## For example when using a spherical shell.
        ## This property is used in the Tumor class.
        self.inner_surface_area = 0

        ## Outer surface area of the compartment. This property is only
        ## relevant when the compartment has a well-defined outer surface.
        ## For example when using a spherical shell.
        ## This property is used in the Tumor class.
        self.outer_surface_area = 0

        ## Thickness of the compartment. This is not the diameter of the
        ## compartment. It simply describes a property that might become 
        ## relevant if we have a specific geometry.
        ## For example when using a spherical shell.
        ## This property is used in the Tumor class.
        self.thickness = 0

        ## Center of the compartment. Treating the compartment as a point mass,
        ## the center coincides with the centroid. 
        ## It is simply the (x,y,z) position of the compartment in 
        ## cartesian coordinates.
        self.center = np.zeros(3)

        ## Each compartment has the property of vascular fraction and
        ## lymphatic fraction, i.e., it describes how much of the volume
        ## of the compartment is composed of blood vessels and how much
        ## corresponds to lymphatics. These information can be utilized to 
        ## compute quantities such as the contact area or the diffusion length 
        ## between compartments.
        self.vascular_fraction = 0

        ## Each compartment has the property of vascular fraction and
        ## lymphatic fraction, i.e., it describes how much of the volume
        ## of the compartment is composed of blood vessels and how much
        ## corresponds to lymphatics. These information can be utilized to 
        ## compute quantities such as the contact area or the diffusion 
        ## length between compartments.
        self.lymphatic_fraction = 0

        ## Contact area related to lymphatic fraction
        self.lymphatic_contact_area = 0

        ## Contact area related to vascular fraction
        self.vascular_contact_area = 0

        ## Diffusion length related to vascular fraction
        self.vascular_diffusion_length = 0

        ## Blood vessel radius: This property is used to compute the
        ## vascular diffusion length and the vascular contact area
        self.blood_vessel_radius = 0

        ## Each compartment object stores a reference to the species list
        ## inside the PBPK class
        self.species = None

        ## Vector of concentrations. This vector has a size equal to the number
        ## of species defined in the PBPK_Driver class. 
        ## To use this vector, one has to update its value using the vector of 
        ## concentrations from the PBPK class and a mapping between local 
        ##and global variables.
        self.concentrations = np.zeros(self.n_species)

        ## These lists represent the template functions for generation, death
        ## and coversion within a compartment. Each compartment has a matrix of
        ## coefficients that make the local reaction functions unique to that
        ## compartment. 

        self.local_generation_functions = []
        self.local_death_functions      = []
        self.local_conversion_functions = []
        self.local_special_functions    = []

        ## Store the state of the activation function
        self.time_vector  = []
        self.state_vector = []


        ## Time local to the compartment
        ## This variable is updated outside of this class
        self.time = 0

        ## We have 5 reaction parameters for each equation
        ## (0) Generation (proportional to concentration)
        ## (1) Death      (proportional to concentration)
        ## (2) Conversion (proportional to the product between concentrations)
        ## (3) Constant rate (independent of concentration)
        ## (4) Special functions
        self.n_reaction_parameters = 5

        self.reaction_labels = ['Generation', 'Death',\
                'Conversion', 'Flat proliferation', 'Additional effects']

        ## The local_reaction_coefficient_matrix has n_species x
        ## n_reaction_parameters entries, where the number of parameters is
        ## 5. Generation, death, conversion and constant rate.
        self.local_reaction_coefficient_matrix = np.zeros((self.n_species,\
                self.n_reaction_parameters))  
        #--------------------------------------------

        ## The local_reaction_coefficient_matrix has n_species (5) x
        ## n_reaction_parameters entries, where the number of parameters is
        ## 5.
        ## We use double indices to indicate the position in the matrix.
        ## The first index indicates the species and the second idicates the
        ## type of reaction.
        self.local_reaction_function_matrix = []
        #self.local_reaction_function_matrix = [[]] * self.n_species

        ## This is a tunning parameter for the transition between the off and
        ## on states of the activation function.
        self.K = 0

        ## Treshold value for the antigen concentration
        self.threshold_value_for_antigen = 0

        ## This boolen variable indicates if the threshold value for the
        ## antigen has been exceeded.
        self.antigen_exceeds_threshold_value = False

        ## This boolen variable indicates if the activation effect is active
        self.activation_function_is_on = False

        ## Current value of the activation function. This function is bounded
        ## between zero and one.
        self.activation_function_value = 0

        ## This boolen variable indicates if the the activation
        ## function has been programmed.
        self.turn_on_activation_function_has_been_scheduled = False

        ## This boolen variable indicates if the shutdown of the activation
        ## function has been programmed.
        self.turn_off_activation_function_has_been_scheduled = False

        ## Time to turn on activation
        self.time_to_turn_on_activation_function = np.inf

        ## Time to shutdown activation
        self.time_to_turn_off_activation_function = np.inf

        ## This boolen variable indicates if the threshold value for the
        ## antigen has ever been exceeded.
        self.has_the_antigen_threshold_value_ever_been_exceeded = False

        self.first_time_the_threshold_value_was_exceeded = np.inf

        self.last_time_activation_function_was_scheduled = np.inf

        self.last_time_activation_function_was_turned_on = np.inf

        self.minimal_duration_for_activation_function = 0

        ## This flag is used to determine if we should start counting 
        self.counting_flag = False

        ## Time to delay the effect of a reaction.
        self.delay_time = 1

        ## Time when the antigen concentration exceeded the threshold value.
        self.last_time_when_antigen_threshold_was_exceeded = np.inf

        ## Time to reach half activation
        self.time_to_reach_half_activation = 0

        ## Real time to reach half activation
        self.real_time_to_reach_half_activation = 0

        self.real_time_to_reach_half_deactivation = 0

        ## Smooth the transition of the activation function
        self.use_smoothing = True

        ## Death boost factor for CD8+ (A)
        self.cd8_death_boost_factor = 0

        ## Virus flag
        self.virus_is_alive = True

        ## Time to reach half activation
        self.directive_to_shutdown_in_progress = False

        '''
        Virus control
        '''
        self.virus_half_max_time    = 0
        self.virus_steepness_factor = 0

        '''
        CD8 (A) death control
        '''
        self.cd8_shutdown_steepness_factor = 0
        self.cd8_time_to_activate_switch = 0
        self.cd8_half_max_time             = 0



#------------------------------------------------------------
#------------------------------------------------------------

        '''IIIIIIIIIIIIII   Initialization   IIIIIIIIIIIIIIII'''

#------------------------------------------------------------
#------------------------------------------------------------
        self.create_dictionaries()
        self.build_generation_death_and_conversion_functions()
        self.test_function_equivalence()
        self.summary()



#------------------------------------------------------------
#------------------------------------------------------------

        '''MMMMMMMMMMMMMMMM Methods MMMMMMMMMMMMMMMMMMMMMMMMM'''

#------------------------------------------------------------
#------------------------------------------------------------
    def create_dictionaries(self):
        '''
        '''
        for k,label in enumerate(self.species_labels):
            self.species_label_to_index[label] = k

#------------------------------------------------------------
#------------------------------------------------------------
    def update_antigen_threshold_state(self):
        '''

        '''

        '''
        Force the system to shut down after first cycle
        '''

        execute_only_one_cycle = True

        condition = execute_only_one_cycle and\
                self.has_the_antigen_threshold_value_ever_been_exceeded

        if condition == True:

            '''
            Time to turn off the activation function.
            '''
            t = self.first_time_the_threshold_value_was_exceeded +\
                    self.delay_time +\
                    self.minimal_duration_for_activation_function 

            if self.activation_function_is_on and (t <= self.time):

                self.time_to_turn_off_activation_function = t

                self.real_time_to_reach_half_deactivation =\
                        self.time_to_turn_off_activation_function +\
                        self.time_to_reach_half_activation

                self.activation_function_is_on         = False 
                self.directive_to_shutdown_in_progress = True

                if self.label == 'Spleen':
                    print 'XXXXXXXX: Turning off activation function'
                    print 'XXXXXXXX: Current time:', self.time

            return
        '''
        ----------------END OF FORCING FUNCTION
        '''

        antigen_index         = self.species_label_to_index['Ag']
        antigen_concentration = self.concentrations[antigen_index]

        if self.threshold_value_for_antigen < antigen_concentration: 
            '''
            Antigen population at a critical level
            '''

            '''
            Has the threshold value ever been exceeded.
            '''
            if not(self.has_the_antigen_threshold_value_ever_been_exceeded):

                self.has_the_antigen_threshold_value_ever_been_exceeded = True
                self.first_time_the_threshold_value_was_exceeded = self.time

            self.antigen_exceeds_threshold_value = True

            '''
            Cancel directive to turn off activation function
            '''
            if self.turn_off_activation_function_has_been_scheduled == True:
                self.turn_off_activation_function_has_been_scheduled = False
                if self.label == 'Spleen':
                    print '>>> Current time:', self.time
                    print '>>> Shutdown instruction has been cancelled'


            '''
            Activation function is off
            Activation has not been scheduled
            '''
            condition =\
                    not(self.activation_function_is_on) and\
                    not(self.turn_on_activation_function_has_been_scheduled)
            if condition == True:

                self.turn_on_activation_function_has_been_scheduled = True

                self.last_time_activation_function_was_scheduled = self.time

                self.time_to_turn_on_activation_function =\
                        self.time + self.delay_time

                '''
                MODIFICATION
                '''
                #self.time_to_turn_on_activation_function = 2.0
                #self.delay_time =\
                        #self.time_to_turn_on_activation_function - self.time

                self.time_to_turn_on_activation_function =\
                        self.time + self.delay_time

                self.real_time_to_reach_half_activation =\
                        self.time_to_turn_on_activation_function +\
                        self.time_to_reach_half_activation

                if self.label == 'Spleen':
                    print '++++++++++++++++++'
                    print '++++++++++++++++++'
                    print 'Inside:', self.label
                    print 'Threshold of activation function reached'
                    print 'Current time: t =', self.time
                    print 'Waiting', self.delay_time, 'units of time'
                    print 'Activation will happen at', self.time_to_turn_on_activation_function

        else:
            '''
            Antigen population below critical
            
            '''
            message    = ''
            wait_time  = np.inf
            print_flag = False

            '''
            Activation function is on
            No turn off event has been scheduled
            '''
            condition =\
                    self.activation_function_is_on and\
                    not(self.turn_off_activation_function_has_been_scheduled)

            if condition == True:

                self.turn_off_activation_function_has_been_scheduled = True

                time_activation_function_has_been_on =\
                        self.time -\
                        self.last_time_activation_function_was_turned_on

                if self.label == 'Spleen':
                    pass
                    #print 'For how long you have been on:',\
                            #time_activation_function_has_been_on 

                remaining_time = np.max((0,\
                        self.minimal_duration_for_activation_function  -\
                        time_activation_function_has_been_on))

                self.time_to_turn_off_activation_function =\
                        self.time + remaining_time


                message = '>>>Activation function was on when the shutdown signal was generated'

                print_flag = self.label == 'Spleen'


            '''
            Turn on activation function has been scheduled
            No turn off event has been scheduled
            '''
            condition =\
                    self.turn_on_activation_function_has_been_scheduled and\
                    not(self.turn_off_activation_function_has_been_scheduled)

            if condition == True:

                self.turn_off_activation_function_has_been_scheduled = True

                self.time_to_turn_off_activation_function =\
                        self.time_to_turn_on_activation_function +\
                        self.minimal_duration_for_activation_function

                message = '>>>Turn on activation function was previously scheduled'

                print_flag = self.label == 'Spleen'

            if print_flag == True:
                print '------------------'
                print '------------------'
                print 'Inside:', self.label
                print message
                print 'Below threshold of activation function reached'
                print 'Current time: t =', self.time
                wait_time =\
                        self.time_to_turn_off_activation_function - self.time
                print 'Function will be shut down at:',\
                        self.time_to_turn_off_activation_function
                print 'Waiting', wait_time, 'units of time'



#------------------------------------------------------------
#------------------------------------------------------------
    def update_activation_function_state(self):
        '''
        First we update the scheduling
        '''

        self.update_antigen_threshold_state()

        delta_t = self.time - self.real_time_to_reach_half_activation
        activation_sign = +1
        activation_step =  0

        if self.directive_to_shutdown_in_progress == True:
            delta_t = self.time - self.real_time_to_reach_half_deactivation
            activation_sign = -1
            activation_step =  1

        '''
        Transition degree
        '''
        activation_value = activation_step +\
                activation_sign /\
                ( 1 + self.use_smoothing * np.exp(-self.K * delta_t) )


        if self.turn_off_activation_function_has_been_scheduled:

            if self.time_to_turn_off_activation_function <= self.time:

                self.activation_function_is_on = False
                self.turn_off_activation_function_has_been_scheduled = False

                if self.label == 'Spleen':
                    print '---------Turning off the activation function'

                activation_value = 0


        if self.turn_on_activation_function_has_been_scheduled:

            if self.time_to_turn_on_activation_function <= self.time:

                self.activation_function_is_on = True
                self.last_time_activation_function_was_turned_on = self.time
                self.turn_on_activation_function_has_been_scheduled = False

                if self.label == 'Spleen':
                    print '+++++++++Turning on the activation function'
                    print '+++++++++Current time:', self.time

        
        '''
        Function is off and not in the process of being shut down.
        '''
        if not(self.activation_function_is_on) and\
                not(self.directive_to_shutdown_in_progress):
            activation_value = 0

        '''
        Update value of activation function
        '''
        self.activation_function_value = activation_value


        self.store_state_of_activation_function()


        
#------------------------------------------------------------
#------------------------------------------------------------
    def store_state_of_activation_function(self):
        '''
        Store the state of the activation function whenever this function is
        called.
        '''

        self.time_vector.append(self.time)
        self.state_vector.append(self.activation_function_value)



#------------------------------------------------------------
#------------------------------------------------------------
    def build_generation_death_and_conversion_functions(self):
        '''
        This function assumes that the generation and death rates are
        constant. 
        The activation value is only used in the functions that need to be
        modulated according to the concentration of antigen.
        '''


        for i in range(self.n_species):
            self.local_reaction_function_matrix.append([])

            if i == 0:
                '''
                We do not consider proliferation of activated CD8+
                '''
                g_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[0] * 0.0

                #d_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        #-s[0]

                def d_fun(s, t = 0, k = 0, activation = 0, use_delay = 0):

                    time_factor = self.cd8_time_to_activate_switch +\
                            self.cd8_half_max_time
                    delta_t = self.time - time_factor
                    argument =\
                            -self.cd8_shutdown_steepness_factor * delta_t
                    death_effectiveness = self.cd8_death_boost_factor *\
                            1./( 1 + np.exp(argument) ) 
                    death_effectiveness *=\
                            (self.cd8_time_to_activate_switch < t)
                    death_effectiveness += 1

                    return -s[0] * death_effectiveness

                '''
                Actived CD8+ population increases proportionally to the 
                ecounter between naive CD8+ and activated DC
                '''
                c_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[1] * s[2] * activation
                k_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        1
                s_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        0

            elif i == 1:
                '''
                CD8+ (N) cells proliferate proportional to the encounters with
                antigen, i.e.,
                [CD8+ (N)] * [Ag]
                This is an overload of the generation function.
                NULLIFIED IN SUITE MODULE
                '''
                g_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[1] * s[4]
                d_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        -s[1]
                '''
                Naive CD8+ population decreases proportionally to the ecounter
                between naive CD8+ and activated DC
                '''
                c_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        - s[1] * s[2] * activation
                k_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        1
                '''
                Naive CD8+ population increases proportionally to the 
                ecounter between naive CD8+ and activated DC
                '''
                s_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[1] * s[2] * activation

            elif i == 2:
                g_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[2]
                d_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        -s[2]
                '''
                Actived DC population increases proportionally to the ecounter
                between naive DC and antigen
                '''
                c_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[3] * np.power(s[4], 1) * activation

                k_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        1
                s_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        0

            elif i == 3:
                g_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        + s[3]
                d_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        -s[3]
                '''
                Naive DC population decreases proportionally to the ecounter
                between naive DC and antigen. This is due to conversion.
                '''
                c_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        -s[3] * np.power(s[4], 1) * activation

                k_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        1
                s_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        0

            elif i == 4:
                '''
                The generation rate of Ag becomes zero once the
                concentration is below a fixed value.
                '''
                def g_fun(s, t = 0, k = 0, activation = 0, use_delay = 0):

                    argument =\
                            -self.virus_steepness_factor *\
                            (t - self.virus_half_max_time)
                    aggressiveness = 1. / ( 1 + np.exp(argument) )
                    '''
                    Aggressiveness factor is off
                    '''
                    aggressiveness = 1

                    if t < 2:
                        return s[4] * aggressiveness
                    else:
                        truncation = (0.001 < s[4]) * s[4]
                        return truncation * aggressiveness

                d_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        -s[4]
                '''
                Antigen (virus) dies proportionally to the encounter between
                virus and activated cd8+ cells
                '''
                c_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        - s[4] * s[0]
                k_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        1
                s_fun = lambda s, t = 0, k = 0, activation = 0, use_delay = 0:\
                        0

            else: 
                '''
                Error
                '''
                print 'Undefined behavior'
                exit()

            '''All functions are in one matrix'''
            self.local_reaction_function_matrix[i].append(g_fun)
            self.local_reaction_function_matrix[i].append(d_fun)
            self.local_reaction_function_matrix[i].append(c_fun)
            self.local_reaction_function_matrix[i].append(k_fun)
            self.local_reaction_function_matrix[i].append(s_fun)
            
            self.local_generation_functions.append(g_fun)
            self.local_death_functions.append(d_fun)
            self.local_conversion_functions.append(c_fun)
            self.local_special_functions.append(s_fun)

#------------------------------------------------------------
#------------------------------------------------------------
    def set_blood_vessel_radius(self, value):
        ''' 
        Set the radius of the blood vessels that make up the vasculature of the
        compartment.
        '''

        self.blood_vessel_radius = value



#------------------------------------------------------------
#------------------------------------------------------------
    def set_vascular_fraction(self, value):
        '''
        This function sets the vascular fraction property and subsequently
        modifies the vascular contact area and the vascular diffusion length.
        Note that these properties are only relevant in the context of the
        connection between a blood tissue and this compartment.
        '''
        self.vascular_fraction = value

        '''
        Update contact area
        V(cylinder) = V(compartment) * vascular_fraction
        V(cylinder) = pi * R^2 * h
        h = V/(pi * R^2)
        A(cylinder,lateral) = 2 * pi * R * h
        '''
        vascular_volume = self.vascular_fraction * self.volume
        cylinder_volume    = vascular_volume 
        vessel_height      =\
                cylinder_volume / (np.pi *\
                np.power(self.blood_vessel_radius, 2))
        lateral_area       = 2 * np.pi *\
                self.blood_vessel_radius * vessel_height

        self.vascular_contact_area = lateral_area


        '''
        Update diffusion length
        '''
        d = (1./np.sqrt(self.vascular_fraction) - 1) *\
                self.blood_vessel_radius / 2

        self.vascular_diffusion_length = d

#------------------------------------------------------------
#------------------------------------------------------------
    def test_function_equivalence(self):
        '''
        Testing function equivalence.
        '''

        s = np.random.rand(self.n_species)
        t = np.random.rand(1)
        tol = 1e-9 

        for s_index in range(self.n_species):
            for i, f in enumerate(self.local_reaction_function_matrix[s_index]):
                new = f(s,t)

                if   i == 0: 
                    old = self.local_generation_functions[s_index](s,t)

                elif i == 1: 
                    old = self.local_death_functions[s_index](s,t)

                elif i == 2: 
                    old = self.local_conversion_functions[s_index](s,t)

                elif i == 3: 
                    old = 1.

                elif i == 4: 
                    old = self.local_special_functions[s_index](s,t)
                else:
                    print 'Unexpected behavior for s_index', s_index,\
                            'and function index', i
                    exit()

                condition = np.abs(old - new) < tol

                if condition == True:
                    pass
                else:
                    print 'Functions do not match'
                    print 'Unexpected behavior for s_index', s_index,\
                            'and function index', i
                    exit()

#------------------------------------------------------------
#------------------------------------------------------------
    def change_parameter_k(self, value):
        '''
        Change the value of the parameter K
        '''
        
        #print '+++Changing K from:', self.K, 'to:', value
        self.K = value


#------------------------------------------------------------
#------------------------------------------------------------
    def summary(self):
        '''
        Summarize the properties of the recently created object.
        '''
        print '+++Compartment object:', self.label, 'created...'
        print '...current volume    :', self.volume

        if self.inner_surface_area != 0: 
            print '...current inner surface area:', self.inner_surface_area
        if self.outer_surface_area != 0: 
            print '...current outer surface area:', self.outer_surface_area
        if self.thickness != 0: 
            print '...current thickness:', self.thickness


#------------------------------------------------------------
#------------------------------------------------------------

    def update_concentrations(self, global_concentrations):
        '''
        Use the global concentration vector to update
        the concentrations local to this compartment.

        We also update the antigen threshold flag.
        '''
        for k, index in enumerate(self.driver_variables):
            self.concentrations[k] = global_concentrations[index]

        '''
        Abort computation if the population of antigen exceeds the limit.
        '''
        if (self.label == 'Spleen') and (100 < self.concentrations[4]):
            print 'Maximum expected population for antigen exceeded'
            exit()

        if (self.label == 'Spleen') and (0 == self.concentrations[4]) and\
                self.virus_is_alive:
            print 'Zero virus at:', self.time
            self.virus_is_alive = False


#------------------------------------------------------------
#------------------------------------------------------------
    def compute_local_reaction(self, dydt):
        '''
        Compute the reaction term for each species that lives within the 
        compartment.
        Note that we need as many functions as species are defined inside the
        compartment.
        Currently we have 5 types of functions:
        (0) Generation
        (1) Death
        (2) Conversion
        (3) Constant generation rate
        (4) Special functions
        '''
        #print self.concentrations

        for s in range(self.n_species):
            temp = 0
            for k, rx_fun in enumerate(self.local_reaction_function_matrix[s]):
                temp += rx_fun(\
                        self.concentrations,\
                        self.time,\
                        self.K,\
                        self.activation_function_value)\
                        *\
                        self.local_reaction_coefficient_matrix[s,k]

            if self.volume == 0:
                print 'Error: Division by zero volume in compartment',\
                        self.label
                exit()

            dydt[self.driver_variables[s]] += temp / self.volume

        

#------------------------------------------------------------
#------------------------------------------------------------
    def old_compute_local_reaction(self, dydt):
        '''
        Compute the reaction term for each species that lives within the 
        compartment.
        Note that we need as many functions as species are defined inside the
        compartment.
        Currently we have 5 types of functions:
        (0) Generation
        (1) Death
        (2) Conversion
        (3) Constant generation rate
        (4) Special functions
        '''
        #print self.concentrations

        for s in range(self.n_species):
            temp = 0

            temp += self.local_generation_functions[s](\
                    self.concentrations,\
                    self.time,\
                    self.K,\
                    self.activation_value) *\
                    self.local_reaction_coefficient_matrix[s,0]

            temp += self.local_death_functions[s](self.concentrations,\
                    self.time, self.K) *\
                    self.local_reaction_coefficient_matrix[s,1]

            temp += self.local_conversion_functions[s](self.concentrations,\
                    self.time, self.K) *\
                    self.local_reaction_coefficient_matrix[s,2]

            temp += self.local_reaction_coefficient_matrix[s,3]

            temp += self.local_special_functions[s](self.concentrations,\
                    self.time, self.K) *\
                    self.local_reaction_coefficient_matrix[s,4]

            if self.volume == 0:
                print 'Error: Division by zero volume in compartment',\
                        self.label
                exit()

            dydt[self.driver_variables[s]] += temp / self.volume

