"""
This tutorial introduces the Echo State Network model using Theano.

An Echo State Networks are a type of recurrent neural model.
While functionally the model has the same form as a recurrent neural model,
the input and recurrent weights are not learnt. Instead they are carefully
sampled, and only the output weights are learned.

.. math::
    h_t = \sigma_h( h_tm1 W_{hh} + u_t W_{uh} + b_h)
    y_t = \sigma_y( h_t W_{hy} + b_y)

An extension of the model considered in this tutorial is the Echo State
Network with Leaky-Integrator Neurons. In this case the formula changes to:

.. math::
    h_t = (1-\alpha) \sigma_h( h_tm1 W_{hh} + u_t W_{uh} + b_h) + \alpha h_tm1
    y_t = \sigma_y( h_t W_{hy} + b_y)

References:
    - scholarpedia: http://www.scholarpedia.org/article/Echo_state_network
"""

import numpy
import pylab
import sys
import time

import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


class ESN:
    def __init__(self,
                 n_in,
                 scale_input_weights=.1,
                 n_out,
                 n_reservoir,
                 leaking_rate=0,
                 spectral_radius=.4,
                 sparcisity=0.05,
                 reservoir_activation=TT.tanh,
                 output_activation=None,
                 inverse_output_activation=None,
                 noise=0.,
                 seed=123):
        """
        Constructs and Echo State Network

        NOTE : Herbert Jaeger's toolbox also offers a few other arguments
            like input shifting and scaling or method of computing the output
            weight matrix (pseudoinverse or Wiener Hopf).
            We have chosen to default to the pseudoinverse, and let the user do
            the input shifting before using the code.
            The code is designed offline training

        :type n_in: int
        :param n_in: number of input units

        :type scale_input_weights: float
        :param scale_input_weights: Element-wise scale of the input weights.
            Each number is uniformly sampled between `-scale_input_weights` and
            `scale_input_weights`.

        :type n_out: int
        :param n_out: number of output units

        :type n_reservoir: int
        :param n_reservoir: number of hidden units

        :type leaking_rate: float
        :param leaking_rate: Amount of leaking for the internal units;
            a leaking_rate of 0 means that the neurons do not perserve any of
            their old value (i.e. leaky-integration neurons are replaced with
            normal neurons)

        :type spectral_radius: float
        :param spectral_radius: During the initialization of the model, the
            recurrent weights are sampled such that their spectral radius is
            equal to this value

        :type sparcisity: float
        :param sparcisity: amount of non-zero recurrent weights given
            between 0 and 1. int(n_reservoir**2 * sparcisity) weights will
            be nonzero

        :type reservoir_activation: theano.Op or function
        :param reservoir_function: non-linearity applied to the internal units;
            the default is tanh which is the standard in the ESN literature

        :type output_activation: theano.Op or function
        :param output_activation: activation function of the output units.
            `None` means linear output units. If you provide some activation
            function, you also have to provide the inverse of this function as
            `inverse_output_activation`.

        :type inverse_output_activation: theano.Op or function
        :param inverse_output_activation: inverse of the output activation
            function.

        :type noise: float
        :param noise: amount of uniform noise to add to the hidden state
            during training

        :type seed: int
        :param seed: seed for random number generator
        """
        dtype = theano.config.floatX
        self.n_in = n_in
        self.n_out = n_out
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.reservoir_activation = reservoir_activation
        self.output_activation = output_activation
        self.inverse_output_activation = inverse_output_activation
        self.rng = numpy.random.RandomState(seed)
        # theano random number generator
        # this is used to add nosie to the reservoir
        self.trng = RandomStreams(self.rng.randint(2**30))

        # number of non zero weights
        non_zero = int(sparcisity*(n_reservoir**2))
        # Generate the weight matrices
        success = 0
        trials = 0
        # directly taken from generate_internal_weights.m (matlab toolbox for ESN released by H. Jaeger)
        while success == 0 :
            try:
                W = numpy.zeros( (n_reservoir**2,))
                W[self.rng.permutation(n_reservoir**2)[:non_zero]] = self.rng.uniform( \
                        low = -1., high = 1., size = (non_zero,) )
                # convert to matrix
                W = W.reshape((n_reservoir, n_reservoir))
                # scale by spectral radius
                maxval = numpy.max(numpy.abs(numpy.linalg.eigvals(W)))
                W = W*sr/maxval
                success = 1
            except:
                trials += 1
                print 'Trial', trials
        print ' Internal weight matrix created'
        # create a shared variable to store our weights
        self.W = theano.shared(W, name = 'W')
        W_in = self.rng.uniform( low = -1., high = 1.,
                size = (n_in, n_reservoir)) * scale_input_weights
        self.W_in = theano.shared(numpy.array(W_in, dtype=dtype), name='W_in')
        self.W_out = theano.shared(numpy.zeros((n_reservoir+n_in, n_out), dtype =dtype), name='W_out')
        # Create the computational graphs
        #
        # .. NOTES :
        #  1. If a variable ends with:
        #      * _t   -> it means the value at time t
        #      * _tmk -> it means the value at time t - k
        #      * _tpk -> it means the value at time t + k
        #
        # Computing the State Matrix
        #
        # Note that instead of doing many vector-matrix products inside
        # scan to compute dot(in_t, self.W_in), you can precompute this
        # outside with a single matrix-matrix product. This is faster,
        # though it consumes more memory.
        # The same holds for the output (which can be computed outside),
        # or sampling noise for the hidden state

        def oneStep(input_projection_t, h_tm1):
            h_t = TT.dot(h_tm1, self.W) + all_t
            if reservoir_activation is not None:
                h_t = activation_function(h_t)
            return h_t*(1.-leaking_rate) + h_tm1*leaking_rate

        input = theano.tensor.matrix('input')
        precomputed_input = theano.dot(input, self.W_in)
        # Note that input.shape[0] is a theano scalar - not a real value yet
        # theano random number generator expects shapes to be either numeric
        # or symbolic, but not a mixture of the two. Because input.shap[0]
        # is symbolic we have to make the whole shape vector symbolic
        precomputed_noise = self.trng.uniform(low = -1., high = 1.,
                size = theano.tensor.stack(input.shape[0],n_reservoir),dtype=dtype) * noise
        # We use 0's to initialize the reservoir
        init_hidden = theano.tensor.constant(numpy.zeros((1,n_reservoir), dtype=dtype),name='0_state')

        hidden_units,updates = theano.scan(oneStep, sequences=[precomputed_input+precomputed_noise],
                outputs_info=[init_hidden], name='hiddenUnits')

        stateMat = theano.tensor.join(1,input, hidden_units.dimshuffle([0,2]))
        self.get_stateMat = theano.function([input], stateMat,updates= updates )

        #### Computing the Output ####


        # Shared variable that keeps track of the previous values of the output
        # initialy we consider 0s as the output values
        h_tm1 = theano.shared( numpy.zeros((1, n_reservoir), dtype=dtype), name='h_tm1')
        # We do same tricks as before; The scan does not do the final
        # hidden units X output matrix because that will imply many
        # vector matrix multiplication; instead this is done outside.
        #  Note : we are exchangeing processor time with memory here
        hidden_units,updates = theano.scan(oneStep, sequences=[precomputed_input],
                outputs_info=[init_hidden], name ='test')
        state = theano.tensor.join(1,  input, hidden_units.dimshuffle([0,2]))
        out = theano.dot(state,self.W_out)
        if output_activation :
            out_t = output_activation(out_t)
        # The dimshuffle is there to remove the added extra dimenstion in the middle
        def eval_oneStep(y_tm1, h_tm1):
            h_t = TT.dot(self.Wm h_tm1.T) +
        self.test_function = theano.function([input], out,updates= updates)


        #theano.printing.pydotprint(self.test_function, 'test_function.png')


    def train(self,data, target = None, wash_out = 0):

        st = time.time()
        stateMat = self.get_stateMat(data)
        print 'Got state matrix. It took %.3f sec'%(time.time()-st)

        print 'Computing the weights (numpy):'
        st2 = time.time()
        # Compute the pseudo-inverse to get the weights
        if self.inverse_output_activation:
            target = self.inverse_output_activation(target)
        weights = numpy.dot(numpy.linalg.pinv(stateMat[wash_out:]),target[wash_out:])
        self.W_out.set_value(weights)
        print 'Done. It took %.3f sec'%(time.time() - st2)
        print 'Total training ', time.time() - st

if __name__ == '__main__':
    # Create the sin .. sin ** 7 dataset

    dx = numpy.sin(numpy.arange(10000)/25.).reshape((10000,1))
    dy = dx[1:] + numpy.random.uniform(low=-.01, high=.01,
                                       size=dx[1:].shape)
    dx = dx[:-1]

    esn = ESN(1,1,600,leaking_rate = 0, sr = .55, noise = 0.001, dtype ='float64')
    esn.train(dx[:6000],dy[:6000], wash_out = 100)
    print 'Getting the output'
    st = time.time()
    out = esn.test_function(dx[6000:])
    print 'It took : %.3f sec' % (time.time() - st)
    out = out.reshape((3999,))
    target = dy[6000:].reshape((3999,))
    print 'Error: ', numpy.mean((out - target)**2)
    pylab.plot(target)


