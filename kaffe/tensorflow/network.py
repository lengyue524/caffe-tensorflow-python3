import numpy as np
import pickle
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.compat.v1.placeholder_with_default(tf.constant(1.0),
                                                                 shape=[],
                                                                 name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        with open(data_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        for op_name in data_dict:
            with tf.compat.v1.variable_scope(op_name, reuse=True):
                # TODO not sure why name mapping does not work
                if 'relu' in op_name:
                    try:
                        var = tf.compat.v1.get_variable(op_name)
                        session.run(var.assign(data_dict[op_name][0]))
                    except ValueError:
                        if not ignore_missing:
                            raise
                else:
                    for param_name, data in data_dict[op_name].iteritems():
                        try:
                            var = tf.compat.v1.get_variable(param_name)
                            session.run(var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.compat.v1.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    def prelu_layer(self, x, weights, biases, name=None):
        """Computes PRelu(x * weight + biases).
        Args:
            x: a 2D tensor.  Dimensions typically: batch, in_units
            weights: a 2D tensor.  Dimensions typically: in_units, out_units
            biases: a 1D tensor.  Dimensions: out_units
            name: A name for the operation (optional).  If not specified
            "nn_prelu_layer" is used.
        Returns:
            A 2-D Tensor computing prelu(matmul(x, weights) + biases).
            Dimensions typically: batch, out_units.
        """
        with ops.name_scope(name, "prelu_layer", [x, weights, biases]) as name:
            x = ops.convert_to_tensor(x, name="x")
            weights = ops.convert_to_tensor(weights, name="weights")
            biases = ops.convert_to_tensor(biases, name="biases")
            xw_plus_b = nn_ops.bias_add(math_ops.matmul(x, weights), biases)
            return self.parametric_relu(xw_plus_b, name=name)

    @layer
    def conv(self,
             inputs,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             prelu=False,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = inputs.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.compat.v1.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(inputs, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, inputs)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            elif prelu:
                output = self.parametric_relu(output, scope=scope)
            return output

    @layer
    def relu(self, x, name):
        return tf.nn.relu(x, name=name)

    @layer
    def prelu(self, x, name):
        return self.parametric_relu(x, name=name)

    def parametric_relu(self, x, scope=None, name="PReLU"):
        """ PReLU.

        Parametric Rectified Linear Unit. Base on:
        https://github.com/tflearn/tflearn/blob/5c23566de6e614a36252a5828d107d001a0d0482/tflearn/activations.py#L188

        Arguments:
            x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
                `int16`, or `int8`.
            name: A name for this activation op (optional).
        Returns:
            A `Tensor` with the same type as `x`.
        """
        # tf.zeros(x.shape, dtype=dtype)
        with tf.compat.v1.variable_scope(scope, default_name=name, values=[x]) as scope:
            # W_init=tf.constant_initializer(0.0)
            # alphas = tf.compat.v1.get_variable(name="alphas", shape=x.get_shape()[-1],
            #                         initializer=W_init,
            #                         dtype=tf.float32)
            alphas = self.make_var(name, x.get_shape()[-1])
            x = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

        x.scope = scope
        x.alphas = alphas
        return x

    @layer
    def max_pool(self, x, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool2d(x,
                                ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding=padding,
                                name=name)

    @layer
    def avg_pool(self, x, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(x,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, x, num_out, name, relu=True, prelu=False):
        with tf.compat.v1.variable_scope(name) as scope:
            input_shape = x.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(x, [-1, dim])
            else:
                feed_in, dim = (x, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            if relu:
                op = tf.nn.relu_layer
            elif prelu:
                op = self.prelu_layer
            else:
                op = tf.compat.v1.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, x, name):
        input_shape = map(lambda v: v.value, x.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                x = tf.squeeze(x, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(x, name=name)

    @layer
    def batch_normalization(self, x, name, scale_offset=True, relu=False, prelu=False):
        # NOTE: Currently, only inference is supported
        with tf.compat.v1.variable_scope(name) as scope:
            shape = [x.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                x,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            elif prelu:
                output = self.parametric_relu(output, name=scope.name)
            return output

    @layer
    def dropout(self, x, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(x, keep, name=name)