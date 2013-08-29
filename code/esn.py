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
    h_t = (1-\gamma) \sigma_h( h_tm1 W_{hh} + u_t W_{uh} + b_h) + \gamma h_tm1
    y_t = \sigma_y( h_t W_{hy} + b_y)

References:
    - scholarpedia: http://www.scholarpedia.org/article/Echo_state_network

.. moduleauthor:: Razvan Pascanu,
"""

import numpy
import pylab
import sys
import time

import theano
import theano.tensor as TT
# Theano comes with its own random generator that is faster (especially on
# GPU). This flag decides if we should rely on the numpy random generator of
# on Theano's
USE_NUMPY_RANDOM_GENERATOR=0
if USE_NUMPY_RANDOM_GENERATOR:
    # This class uses numpy's number generator (slow)
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    # Custom random generator written for Theano (faster)
    from theano.sandbox.mrg_rng import MRG_RandomStreams as RandomStreams


class ESN:
    def __init__(self,
                 n_in,
                 input_scaling,
                 input_shifting,
                 n_out,
                 target_scaling,
                 target_shifting,
                 feedback_scaling,
                 n_reservoir,
                 leaking_rate,
                 spectral_radius,
                 sparsity,
                 bias_scale,
                 reservoir_activation=TT.tanh,
                 output_activation=None,
                 inverse_output_activation=None,
                 training_noise=0.,
                 seed=123):
        """
        Constructs and Echo State Network

        NOTE : One has usually the option of picking how to compute the
            output weight matrix (either by calling pseudoinverse or using a
            Wiener Hopf approximation). See for e.g. Herbert Jaeger's
            toolbox at
            www.faculty.jacobs-university.de/hjaeger/pubs/freqGen.zip
            We have chosen to default to pseudoinverse. Also the code is
            designed for offline training.

        :type n_in: int
        :param n_in: number of input units

        :type input_scaling: float, numpy.ndarray
        :param input_scaling: Either a float or a vector of floats (of same
            dimensionality as the input). The input is multiplied by this
            value before being fed into the model. The formula used is:
                input * input_scaling + input_shifting

        :type input_shifting: float, numpy.ndarray
        :param input_shifting: Either a float or a vector of floats (of same
            dimensionality as the input). After scaling, the input is
            shifted by this value. The formula used is:
                input * input_scaling + input_shifting

        :type n_out: int
        :param n_out: number of output units

        :type target_scaling: float, numpy.ndarray
        :param target_scaling: Either a float or a vector of floats (of same
            dimensionality as the target). The target is scaled by this
            value before being used to compute the error. Formula used is:
                target * target_scaling + target_shifting

        :type target_shifting: float, numpy.ndarray
        :param target_shifting: Either a float or a vector of floats (of
            same dimensionality of the target). After scaling, the target is
            shifted by this value. The formula used is:
                target * target_scaling + target_shifting

        :type feedback_scaling: float, numpy.ndarray
        :param feedback_scaling: Either a float or a vector of floats (of
            same dimensionality of the target). The predictions of the
            models are scaled by this value before being fed back into the
            reservoir.

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

        :type sparsity: float
        :param sparsity: amount of non-zero recurrent weights given
            between 0 and 1. int(n_reservoir**2 * sparsity) weights will
            be nonzero

        :type bias_scale: positive float
        :param bias_scale: biases of the hidden state will be sampled from a
            uniform distribution [-bias_scale, bias_scale]

        :type reservoir_activation: theano.Op or function
        :param reservoir_function: non-linearity applied to the internal units;
            the default is tanh which is the standard in the ESN literature.
            Note that this function should receive as input a symbolic
            expression and output a symbolic expression

        :type output_activation: theano.Op or function
        :param output_activation: activation function of the output units.
            `None` means linear output units. If you provide some activation
            function, you also have to provide the inverse of this function as
            `inverse_output_activation`. Note that this function should
            receive as input a symbolic expression and output a symbolic
            expression

        :type inverse_output_activation: function
        :param inverse_output_activation: inverse of the output activation
            function. Note that this function should receive as input a
            numeric expression and return a numerci expression. `None`
            implies not using any function.

        :type noise: float
        :param noise: amount of uniform noise to add to the hidden state
            during training

        :type seed: int
        :param seed: seed for random number generator
        """
        # For simplicity we construct a shortcut to theano.config.floatX
        floatX = theano.config.floatX
        # Attach the different arguments of the class to self, for future
        # reference
        self.n_in = n_in
        assert n_in > 0, "n_in has to be positive"
        self.input_scaling = input_scaling
        self.input_shifting = input_shifting
        self.n_out = n_out
        assert n_out > 0, "n_out has to be positive"
        self.target_scaling = target_scaling
        self.target_shifting = target_shifting
        self.feedback_scaling = feedback_scaling
        self.n_reservoir = n_reservoir
        assert n_reservoir > 0, "n_reservoir has to be positive"
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        assert spectral_radius >0, "spectral_radius has to be positive"
        self.sparsity = sparsity
        self.bias_scale = bias_scale
        assert bias_scale > 0, "bias_scale has to be positive"
        self.reservoir_activation = reservoir_activation
        self.output_activation = output_activation
        self.inverse_output_activation = inverse_output_activation
        # Construct the random number generator used to sample the weights
        self.rng = numpy.random.RandomState(seed)
        # Construct a symbolic random number generator
        # this is used to sample noise to be added to the reservoir
        self.trng = RandomStreams(self.rng.randint(2**30))

        # number of non zero weights
        n_non_zero = int(sparsity*(n_reservoir**2))
        # Generate the recurrent weight matrix
        # Computing the eigenvalues of the randomly sampled matrix can fail,
        # therefore we allow a maximum of 10 trials
        success = 0
        trials = 0
        # code inspired by generate_internal_weights.m (matlab toolbox
        # for ESN released by H. Jaeger)
        while success == 0 and trials < 10:
            try:
                W_hh = numpy.zeros( (n_reservoir**2,))
                non_zero = self.rng.permutation(n_reservoir**2)[:n_non_zero]
                W_hh[non_zero] = self.rng.uniform(low=-1.,
                                                  high=1.,
                                                  size=(non_zero,))
                # convert to matrix
                W_hh = W_hh.reshape((n_reservoir, n_reservoir))
                # scale by spectral radius
                maxval = numpy.max(numpy.abs(numpy.linalg.eigvals(W_hh)))
                W_hh = W_hh * spectral_radius/maxval
                success = 1
            except:
                trials += 1
                print 'Trial', trials
        if success == 0:
            raise Exception("Could not sample a recurrent weight matrix."
                            "Try to make the matrix less sparse")
        self.W_hh = theano.shared(W_hh, name = 'W_hh')
        print 'Construct non-recurrent weights'
        # create a shared variable to store our weights
        W_uh = self.rng.uniform(low=-1., high=1., size=(n_in, n_reservoir))
        self.W_uh = theano.shared(W_uh.astype(floatX), name='W_uh')
        W_uy = numpy.zeros((n_in, n_out), dtype=floatX)
        self.W_uy = theano.shared(W_uy.astype(floatX), name='W_uy')
        W_hy = numpy.zeros((n_reservoir, n_out), dtype=floatX)
        self.W_hy = theano.shared(W_hy, name='W_hy')
        W_yh = self.rng.uniform(low=-1., high=1., size=(n_out, n_reservoir))
        self.W_yh = theano.shared(W_yh, name='W_yh')
        b_h = self.rng.uniform(low=-1, high=1, size=(n_reservoir,)) * bias_scale
        self.b_h = theano.shared(b_h.astype(floatX), name='b_h')
        b_y = numpy.zeros((n_outs,), dtype=floatX)
        self.b_y = theano.shared(b_y.astype(floatX), name='b_y')


        # .. NOTE :
        #
        # Inside the loop of the recurrent network we need to compute:
        #   tanh(dot(u_t, W_uh) +
        #        dot(y_tm1, W_yh) +
        #        dot(h_tm1, W_hh) +
        #        b)
        # Two of the matrix-vector products [dot(u_t, W_uh) and
        # dot(y_tm1, W_yh)] do not the recurrently computed state h_t.
        # We have all required u_t (the sequence of inputs steps) and
        # y_tm1 (the target when we do teacher forcing). Therefore we can
        # replace these many matrix-vector products by a two
        # matrix-matrix products outside of the loop.
        # While this will give a considerable speed-up, it does imply
        # consuming more memory.

        # u, y_tm1 are the whole input and target sequence
        # They take the form of a matrix, where the first dimension goes
        # over time, while the second dimension is the size of each input
        # step
        # y_tm1 is the target shifted by -1 in time. I.e. the first element
        # y_tm1[0] is 0. The second one y_tm1[1] equals the target for step
        # 0 (y[0]). y_tm1[2] corresponds to y[1] and so forth.
        # This is because the feedback connections have a one time step
        # delay, i.e. we feedback the predicted output not the one we need
        # to predict.
        u = TT.matrix('input')
        y_tm1 = TT.matrix('target')
        u_scaled = u * input_scaling + input_shifting
        y_tm1_scaled = y_tm1 * target_scaling + target_shifting
        precomputed_u = TT.dot(u_scaled, self.W_uh)
        precomputed_y = TT.dot(y_tm1_scaled * feedback_scaling, self.W_yh)
        # The same way we can precompute the projection of the input in the
        # latend space, we can sample in one go the noise we need to add to
        # the hidden state
        # Note that input.shape[0] is a theano scalar - not a real value yet
        # theano random number generator expects shapes to be either numeric
        # or symbolic, but not a mixture of the two. Because input.shape[0]
        # is symbolic we have to make the whole shape vector symbolic
        shape_noise = TT.stack(input.shape[0], n_reservoir)
        precomputed_noise = self.trng.uniform(
            low=-1., high=1., size=shape_noise, dtype=floatX) * noise
        # Now we can sum together all these pre-computed values
        u_y_b_noise_to_h = precomputed_u + \
                           precomputed_y + \
                           self.b_h + \
                           precomputed_noise


        # The way looping is done in Theano is by using the scan node.
        # We strongly encourage you to read the tutorial on Theano and on
        # scan in order to get a better understanding of this code.

        # Scan asks for a python function (or lambda expression) that
        # describe the computations that need to be carried out by each step
        # of the loop. This function gets as input the right slices of the
        # sequences over which we loop (and states that we compute
        # recursively). The order of the arguments is important. We always
        # have first the sequences, followed by the states followed by
        # auxiliary variables (parameters) from which scan does not take a
        # slice. In general, if those parameters are available in the scope
        # of the function you do not need to pass them explicitly.
        def oneStep_training(u_y_b_noise_to_h_t, h_tm1):
            """
            Describes one step of the loop (at training time)

            :type u_y_b_noise_to_h_t: theano tensor
            :param u_y_b_noise_to_h_t: theano tensor representing a slice of
                the input projection to the hidden state (the sum of
                precomputed dot products of input - W_uh weight matrix,
                previous output - W_yh feedback weight matrix and the
                pregenerated noise and bias).
                This variable corresponds (due to convention and how the
                scan node is called below) to a sequence.

            :type h_tm1: theano tensor
            :param h_tm1: theano tensor representing the previous state of
                the hidden layer
            """
            # compute the new (linear) value of the hidden state
            h_t = TT.dot(h_tm1, self.W_hh) + u_y_b_noise_to_h_t
            # apply the non-linearity if one is defined
            if reservoir_activation is not None:
                h_t = reservoir_function(h_t)
            # apply the leaky-integration formula
            return h_t*(1.-leaking_rate) + h_tm1*leaking_rate

        # Scan asks the user to provide a initial state, from which we start
        # the recurrent formula. In our case we provide 0s.
        init_hidden = tt.constant(
            numpy.zeros((1,n_reservoir), dtype=floatx),name='0_state')

        # We construct the main loop by saying which function it needs to
        # implement at each time step, what are the sequences it has to loop
        # over (in our case just one), and what are the initial states for
        # the different recursively computed outputs.
        hidden_units, updates = theano.scan(
            oneStep_training,
            sequences=u_y_b_noise_to_h,
            outputs_info=init_hidden,
            name='training_rnn_loop')

        # Compile the theano function that will give us the hidden states
        # which we need for computing the output weights
        self.get_hidden_state = theano.function([u, y_tm1],
                                                hidden_units,
                                                updates=updates)

        # Generating the evaluation graph

        # We need to redefine the `oneStep_training` function (the one scan
        # uses to know what it needs to do). One reason is that compared to
        # training time, we do not have the previous outputs generated by
        # the model before hand. This becomes now an output computed
        # recursively by the model. Also we do not add noise to the
        # reservoir at evaluation time.
        input_projection = precomputed_u + self.b_h

        def oneStep_evaluation(u_b_to_h_t, u_b_to_y_t,  h_tm1, y_tm1):
            """
            Describes one step of the loop (at training time)

            :type u_b_to_h_t: theano tensor
            :param u_b_to_h_t: theano tensor representing a slice of
                the input projected to the latent space (the sum of
                precomputed dot product of input - W_uh weight matrix and
                the bias). This variable corresponds (due to convention
                and how the scan node is called below) to a sequence.

            :type u_to_y_t: theano tensor
            :param u_to_y_t: theano tensor representing a slice of
                the precomputed dot(u_t, W_uy) + b_y
                This variable corresponds (due to convention and how the
                scan node is called below) to a sequence.

            :type h_tm1: theano tensor
            :param h_tm1: theano tensor representing the previous state of
                the hidden layer

            :type y_tm1: theano tensor
            :param y_tm1: theano tensor representing the previous made
                prediction
            """
            h_t = TT.dot(h_tm1, selef.W_hh) + \
                  TT.dot(y_tm1 * feedbackScaling, self.W_yh) + \
                  u_b_to_h_t
            if reservoir_activation is not None:
                h_t = reservoir_function(h_t)
            lh_t = h_t * (1.-leaking_rate) + h_tm1 * leaking_rate
            y_t = TT.dot(lh_t, self.W_hy) + u_b_to_y_t
            if output_activation is not None:
                y_t = output_activation(y_t)
            return h_t, y_t

        # The initial state of the previously predicted output (set to 0s)
        init_out = TT.constant(
            numpy.zeros((1,n_reservoir), dtype=floatx),name='0_state')

        [_, output], updates = theano.scan(
            oneStep_evaluation,
            sequences=[u_b_to_h, u_b_to_y]
            outputs_info=[init_hidden, init_out],
            name='rvaluation_rnn_loop')
        self.test_function = theano.function([u], output, updates=updates)


    def train(self,data, target, wash_out = 0):
        """
        Given data, this function learns the output weights (and bias).

        :type data: numpy.ndarray
        :param data: numpy array containing the input sequence for training
            the model

        :type target: numpy.ndarray
        :param target: numpy array containing the target the model has to
            learn

        :type wash_out: int
        :param wash_out: how many first entry of the inputs one should
            ignore (burn in period required due to the initial state of the
            hidden units)

        .. NOTE :
            The target has to be one entry longer then the data. The first
            entry (target[0]) should be zero. target[1] should be the target
            corresponding to data[0].
        """

        start_time = time.time()
        hidden_state = self.get_hidden_state(data, target[:1])
        bias = numpy.ones((data.shape[0],), dtype=theano.config.floatX)
        state_mat = numpy.concatenate((hidden_state, data, bias))
        print 'Got state matrix. It took %.3f sec'%(time.time() - start_time)

        # For now we will rely on numpy linear algebra package to compute the
        # pseudoinverse. A similar Op exists in theano linear algeba package
        # though for now it just calls nunpy
        print 'Computing the weights (in numpy):'
        computing_weights = time.time()
        # Compute the pseudo-inverse to get the weights
        if self.inverse_output_activation:
            target = self.inverse_output_activation(target)
        weights = numpy.dot(
            numpy.linalg.pinv(stateMat[wash_out:]),
            target[wash_out+1:])
        # get back the input-output, hidden-output and bias (slices)
        self.W_hy.set_value(weights[:self.n_reservoir])
        self.W_uy.set_value(weights[self.n_reservoir:self.n_reservoir +
                                    self.n_in])
        self.b_y.set_value(weights[self.n_reservoir + self.n_in:])
        print 'Done. It took %.3f sec'%(time.time() - computing_weights)
        print 'Total training time %.3f sec'% (time.time() - start_time)

if __name__ == '__main__':
    # Frequency generator task
    # This is the same task as the one presented in the scholarpedia article
    # http://www.scholarpedia.org/article/Echo_state_network
    # Given a slow signal generate a fast version
    # Step 1. Generate data on the fly [most work]
    # Step 2. Construct model
    # Step 3. Train Model
    # Step 4. Evaluate the model

    dx = numpy.sin(numpy.arange(10000)/25.).reshape((10000,1))
    dy = dx[1:] + numpy.random.uniform(low=-.01, high=.01,
                                       size=dx[1:].shape)
    dx = dx[:-1]

    esn = ESN(1,1,600,leaking_rate = 0, sr = .55, noise = 0.001)
    esn.train(dx[:6000],dy[:6000], wash_out = 100)
    print 'Getting the output'
    st = time.time()
    out = esn.test_function(dx[6000:])
    print 'It took : %.3f sec' % (time.time() - st)
    out = out.reshape((3999,))
    target = dy[6000:].reshape((3999,))
    print 'Error: ', numpy.mean((out - target)**2)
    pylab.plot(target)


