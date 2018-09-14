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
#rc('text', usetex='True')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------SPECIES CLASS-----------------------
#------------------------------------------------------------
#------------------------------------------------------------

class Species:
    '''This class aims to describe a generic substance that is transported
    through a network of compartments. Current properties under consideration
    are:
    (0) P Partition coefficient between compartments
    (1) D Diffusion coefficient between tissues
    (2) A Contact area between tissues
    (3) L Diffusion length between tissues
    '''

    def __init__(self,\
            parent             = None,
            label              = ''):
        '''
        Each species possesses a label and each object is aware of which
        tissues and compartments exist within the PBPK model.  We also require
        the compartment labels because properties such as the partition
        coefficient apply only to compartments and not tissues.
        '''

        ## Parent object. This object provides basic information such as
        ## mappings.
        self.parent              = parent

        ## Name of the given species
        self.label               = label

        ## Number of tissues
        self.n_tissues           = self.parent.n_tissues

        ## Number of compartments
        self.n_compartments      = self.parent.n_compartments

        ## This property is only relevant in the context of perturbations.
        ## That is for the sensitivity analysis.
        self.K                   = 1.0

        ## This property is only relevant in the context of perturbations.
        ## That is for the sensitivity analysis.
        ## See the multiplicative_factor() function and the 
        ## perturb_overall_transport_coefficient() function.
        self.target_i_index      = -1


        ## This property is only relevant in the context of perturbations.
        ## That is for the sensitivity analysis.
        ## See the multiplicative_factor() function and the 
        ## perturb_overall_transport_coefficient() function.
        self.target_j_index      = -1


        self.attribute_labels     = (\
                'partition_coefficient',\
                'diffusion_coefficient',\
                'contact_area'\
                'diffusion_length'\
                )

        self.partition_coefficient = None
        self.contact_area          = None
        self.diffusion_coefficient = None
        self.diffusion_length      = None

        '''MMMMMMMMMMMMMMMM Methods MMMMMMMMMMMMMMMMMMMMMMMMM'''

        '''
        Injection related properties
        '''
        self.injection_compartment_indices = []

        #Initialize data
        self.initialize_data()

#------------------------------------------------------------
#------------------------------------------------------------

    def initialize_data(self):
        '''
        Populate all the arrays that correspond to physical parameters with 
        the constant one.
        '''

        '''
        NOTE: The partition coefficient is a property of compartments, not of
        tissues. Hence the partition_coefficient matrix only accommodates
        compartments.
        '''
        self.partition_coefficient = np.ones((self.n_compartments,\
                self.n_compartments))
        self.contact_area          = np.ones((self.n_tissues, self.n_tissues))
        self.diffusion_coefficient = np.ones((self.n_tissues, self.n_tissues))
        self.diffusion_length      = np.ones((self.n_tissues, self.n_tissues))


#------------------------------------------------------------
#------------------------------------------------------------

    def set_all_properties(self, diffusion_coefficient, diffusion_length, \
            contact_area, partition_coefficient):
        '''
        Use the data provided by the user to populate the properties of
        each material. Note that we do not assume that the given matrices have
        the same size as the matrices they try to populate. We fill them entry
        by entry to avoid replacing a matrix with another one of a different
        size.
        '''

        '''
        Tissue properties
        We copy entry by entry and avoid copying a reference.
        '''

        np.copyto(self.diffusion_coefficient, diffusion_coefficient)
        np.copyto(self.diffusion_length, diffusion_length)
        np.copyto(self.contact_area, contact_area)

        '''This is a compartment property'''
        np.copyto(self.partition_coefficient, partition_coefficient)

#------------------------------------------------------------
#------------------------------------------------------------


    def compute_effective_partition(self, i, j, choice = 'max'):
        '''
        Compute or retrieve the partition coefficient between 
        two compartments (not tissues).


        This function allows us to choose among different implementations of
        the effect of the partition coefficient in the flux.
        ''' 

        p = self.partition_coefficient[i,j]
        p0, p1 = (0,0)
        if choice == 'max':
            p1 = max(p, 1)
            p0 = max(1./p, 1)
        elif choice == 'min': 
            p1 = min(p, 1)
            p0 = min(1./p, 1)
        elif choice == 'direct': 
            p1 = p
            p0 = 1
        elif choice == 'inverse': 
            p1 = 1
            p0 = 1./p
        else:
            error('Undefined partition function type')

        return (p0,p1)

#------------------------------------------------------------
#------------------------------------------------------------

    def apply_partition_coefficient(self, i, j, concentration_i,\
            concentration_j):
        '''
        Apply the effect of the partition coefficient to the diffusion term.
        We assume that the material diffuses from i ---> j

        This function is called inside the compute_diffusion()
        function. Hence, the indices i and j make reference to the 
        tissues under consideration, not compartments. Internally this 
        function computes the corresponding compartment indices.
        '''

        i_c = self.parent.tissue_index_to_compartment_index[i]

        j_c = self.parent.tissue_index_to_compartment_index[j]

        p0,p1 = self.compute_effective_partition(i_c, j_c)

        '''
        If one of the compartments belongs to the list of injection sites,
        then we only allow flow from the injection site to the adjacent
        compartments, i.e., unidirectional flow.
        '''
        one_compartment_is_an_injection_site =\
                (i_c in self.injection_compartment_indices) or\
                (j_c in self.injection_compartment_indices)
        if one_compartment_is_an_injection_site: 
            if concentration_i < concentration_j: 
                return 0

        return p1*concentration_j - p0*concentration_i

#------------------------------------------------------------
#------------------------------------------------------------

    def multiplicative_factor(self, i, j):
        '''
        Control directly the transport coefficient
        '''
        if (i == self.target_i_index) and (j == self.target_j_index):
            return self.K
        else:
            return 1.0
        
#------------------------------------------------------------
#------------------------------------------------------------

    def compute_diffusion_factor(self, i, j):
        '''
        Compute the diffusion multiplier of a species from i ---> j
        '''
        #flow = -1. * self.multiplicative_factor(i,j) *
        #flow = -1. * self.K *

        '''We make sure the diffusion length is positive'''
        if self.diffusion_length[i,j] <= 1e-9:
            print '*** Error:', 'compute_diffusion_factor() in class Species'
            print 'Diffusion length is zero!'
            print 'Tissues:', self.parent.tissue_labels[i],\
                    '<--->', self.parent.tissue_labels[j]
            print 'Terminating simulation'
            exit()

        flow = -1. *\
                self.diffusion_coefficient[i,j] * \
                self.contact_area[i,j] / self.diffusion_length[i,j]

        return flow

#------------------------------------------------------------
#------------------------------------------------------------

    def compute_diffusion(self, i, j, c_i, c_j):
        '''
        Compute the diffusion term of a species from i ---> j
        '''
        flow = self.compute_diffusion_factor(i, j) * \
                self.apply_partition_coefficient(i, j, c_i, c_j)
        return flow


#------------------------------------------------------------
#------------------------------------------------------------

    def symmetric_modification_using_label(\
            self, attribute, x_label, y_label, value,\
            show_final_state = False):
        '''
        Set the (i,j) entry of a given attribute matrix to a given value.
        The entry (j,i) is also set to the same value.
        This function assumes the property refers to a tissue.
        Applications: E.g., the diffusion length.
        '''
        x_index = self.parent.tissue_label_to_index[x_label]
        y_index = self.parent.tissue_label_to_index[y_label]
        getattr(self, attribute)[x_index, y_index] = value
        getattr(self, attribute)[y_index, x_index] = value

        if show_final_state == True: 
            print getattr(self, attribute)


#------------------------------------------------------------
#------------------------------------------------------------

    def symmetric_modification_using_index(\
            self, attribute, x_index, y_index, value,\
            show_final_state = False):
        '''
        Set the (i,j) entry of a given attribute matrix to a given value.
        The entry (j,i) is also set to the same value.
        This function assumes the property refers to a tissue.
        Applications: E.g., the diffusion length.
        '''
        getattr(self, attribute)[x_index, y_index] = value
        getattr(self, attribute)[y_index, x_index] = value

        if show_final_state == True: 
            print getattr(self, attribute)

#------------------------------------------------------------
#------------------------------------------------------------

    def i_symmetric_modification(self, attribute, x_label, y_label, value,\
            show_final_state = False):
        '''
        Set the (i, j) entry of a given attribute matrix to a given value.
        The entry (j, i) is set to ( 1. / value )
        This function assume the property refers to a compartment.
        Applications: E.g., the partition coefficient matrix.
        '''
        x_index = self.parent.compartment_label_to_index[x_label]
        y_index = self.parent.compartment_label_to_index[y_label]
        getattr(self, attribute)[x_index, y_index] = value
        getattr(self, attribute)[y_index, x_index] = 1. / value

        if show_final_state == True: 
            print getattr(self, attribute)
