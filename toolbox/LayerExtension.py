import tensorflow as tf
import tensorlayer as tl


class ConvRNNLayer(tl.layers.Layer):
    """
    The :class:`RNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_shape : tuple, the shape of each cell width*height
    filter_size : tuple, the size of filter width*height
    cell_fn : a TensorFlow's core Convolutional RNN cell as follow.
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    feature_map : a int
        The number of feature map in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    return_last : boolen
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    """
    def __init__(
        self,
        layer = None,
        cell_shape = None,
        feature_map = 1,
        filter_size = (3, 3),
        cell_fn = None,
        cell_init_args = {},
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        n_steps=5,
        initial_state = None,
        return_last = False,
        return_seq_2d = False,
        name='convlstm_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate RNNLayer %s: feature_map:%d, n_steps:%d, "
              "in_dim:%d %s, cell_fn:%s " % (self.name, feature_map,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 5 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(5)
        except:
            raise Exception("RNN : Input dimension should be rank 5 : [batch_size, n_steps, input_x, "
                            "input_y, feature_map]")



        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        self.cell = cell = cell_fn(shape=cell_shape, filter_size=filter_size, num_features=feature_map)
        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 1.2.3
        state = self.initial_state
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :, :, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

        print(" n_params : %d" % (len(rnn_variables)))

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = outputs[-1]
            # print 'Hello', self.outputs.shape
        else:
            if return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, cell_shape[0]*cell_shape[1]*feature_map])
            else:
                # <akara>: stack more RNN layer after that
                # 5D Tensor [n_example/n_steps, n_steps, n_hidden]
                self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_steps, cell_shape[0],
                                                                  cell_shape[1], feature_map])

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )




        ## fixed
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        ## theta_layer
        # print offset_layer.all_params

        offset_params = [osparam for osparam in offset_layer.all_params if osparam not in layer.all_params]
        offset_layers = [oslayer for oslayer in offset_layer.all_layers if oslayer not in layer.all_layers]

        self.all_params.extend(offset_params)
        self.all_layers.extend(offset_layers)
        self.all_drop.update(offset_layer.all_drop)

        # print 'haha', self.all_params
        ## this layer
        self.all_layers.extend([self.outputs])
        self.all_params.extend([W, b])

