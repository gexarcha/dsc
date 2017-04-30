#
#  Lincense: Academic Free License (AFL) v3.0
#
"""
"""



import numpy as np
from mpi4py import MPI


import dsc.utils.parallel as parallel

from dsc.utils.datalog import dlog


# EM Class

class EM():
    """ This class drives the EM algorithm. 

    """
    def __init__(self, model=None, anneal=None, data=None, 
                  lparams=None, mpi_comm=None):
        """ Create a new EM instance 
        
        :param model:  the actual model to train
        :type  model:  :class:`Model` instance
        :param anneal: an annealing schedule to use
        :type  anneal: :class:`annealing.Annealing` instance
        :param data:   Training data in a dictionary. The required content is model dependent,
                but usually data['y'] should contain the trainig data.
        :type  data:   dict
        :param lparam: Inital values for all the model parameters to learn
        :type  lparam: dict

        All these parameters can be changed after initialization by assigning a value
        to the corresponding attributes.
        """
        self.model = model;
        self.anneal = anneal
        self.data = data
        self.lparams = lparams
        self.mpi_comm = mpi_comm

    def step(self):
        """ Execute a single EM-Step """
        model = self.model
        anneal = self.anneal
        my_data = self.data
        model_params = self.lparams

        # Do an complete EM-step
        new_model_params = model.step(anneal, model_params, my_data)

    def run(self, verbose=False):
        """ Run a complete cooling-cycle 

        When *verbose* is True a progress message is printed for every step
        via :func:`dlog.progress(...)`
        """
        model = self.model
        anneal = self.anneal
        my_data = self.data
        model_params = self.lparams

        while not anneal.finished:
            # Progress message
            if verbose:
                dlog.progress("EM step %d of %d" % (anneal['step']+1, anneal['max_step']), anneal['position'])

            # Do E and M step
            new_model_params = model.step(anneal, model_params, my_data)
            
            # Calculate the gain so that dynamic annealing schemes can be implemented
            gain = model.gain(model_params, new_model_params)

            anneal.next(gain)
            if anneal.accept:
                model_params = new_model_params

            self.lparams = model_params

