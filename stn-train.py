import toolbox.NetFlow as nf
from toolbox import DataProvider
from toolbox import LayerExtension
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse

sess = tf.InteractiveSession()


def get_arguments():

    parser = argparse.ArgumentParser(description='Train a STN\
                                     for mobile traffic forecasting')
    parser.add_argument('--datadir',
                        type=str,
                        default='./data/',
                        help='this is the directory of the training samples')
    parser.add_argument('--batchsize',
                        type=int,
                        default=100,
                        help='The batch size of training examples')
    parser.add_argument('--epoch',
                        type=int,
                        default=50,
                        help='The number of epoches.')
    parser.add_argument('--save_model',
                        type=str,
                        default='stn.npz',
                        help='Save the learnt model: \
                            0 -- not to save the learnt model parameters;\
                            n (n>0) -- to save the model params every n steps;\
                            -1 -- only save the learnt model params at the end of training.\
                             n = 1 by default')
    parser.add_argument('--model_path',
                        type=str,
                        default='./',
                        help='Saved model path')
    parser.add_argument('--model_name',
                        type=str,
                        default='stn',
                        help='Saved model name')
    parser.add_argument('--pre_model_path',
                        type=str,
                        default=None,
                        help='Pretrained model path')
    parser.add_argument('--pre_model_name',
                        type=str,
                        default=None,
                        help='Pretrained model name')
    parser.add_argument('--mean',
                        type=float,
                        default=0,
                        help='mean value for data normalisation: \
                            (data-mean)/std')
    parser.add_argument('--std',
                        type=float,
                        default=1,
                        help='standard deviation value for data normalisation: \
                            (data-mean)/std')
    parser.add_argument('--observations',
                        type=int,
                        default=12,
                        help='temporal length of input')
    parser.add_argument('--lr',
                        type=int,
                        default=0.001,
                        help='learning rate of the model')
    parser.add_argument('--framebatch',
                        type=int,
                        default=1,
                        help="maximum frames selected in one mini-batch")
    parser.add_argument('--input_x',
                        type=int,
                        default=11,
                        help="spatial length of input of x axis")
    parser.add_argument('--input_y',
                        type=int,
                        default=11,
                        help="spatial length of input of y axis")
    parser.add_argument('--pad',
                        type=tuple,
                        default=(5, 5),
                        help="2-element or None, the size of padding")
    parser.add_argument('--pad_value',
                        type=int,
                        default=0,
                        help="the value of padding")
    parser.add_argument('--prediction_gap',
                        type=int,
                        default=1,
                        help="the distance between the last input frame and output frame")
    parser.add_argument('--stride',
                        type=tuple,
                        default=(1, 1),
                        help="2-element tuple, the stride of slicing the original snapshot")
    parser.add_argument('--ouroboros_e',
                        type=int,
                        default=0,
                        help="The epoch for the ouroboros training scheme")
    return parser.parse_args()

args = get_arguments()


def load_dataset():

    tra_set = np.load(args.datadir + 'milan_tra.npy')
    val_set = np.load(args.datadir + 'milan_val.npy')
    test_set = np.load(args.datadir + 'milan_test.npy')
    print('training set:', tra_set.shape)
    print('validation set:', val_set.shape)
    print('test set:', test_set.shape)

    return tra_set, val_set, test_set

tra_set, val_set, test_set = load_dataset()


framebatch = args.framebatch
stride = args.stride
input_size = (args.input_x, args.input_y, args.input_x, args.observations)
prediction_gap = args.prediction_gap
batchsize = args.batchsize
pad = args.pad
pad_value = args.pad_value
num_epochs = args.epoch
learning_rate = args.lr
shuffle = True
flatten = True
output_size = (1,1,1)

tra_kwag = {
    'inputs':tra_set,
    'framebatch': framebatch,
    'mean': args.mean,
    'std': args.std,
    'norm_tar': True}

val_kwag = {
    'inputs': val_set,
    'framebatch': framebatch,
    'mean': args.mean,
    'std': args.std,
    'norm_tar': True}

test_kwag = {
    'inputs': test_set,
    'framebatch': framebatch,
    'mean': args.mean,
    'std': args.std,
    'norm_tar': True}

tra_provider = DataProvider.Provider(stride = stride, input_size = input_size,
                           output_size = output_size, prediction_gap = prediction_gap,
                           batchsize = batchsize, pad = pad, pad_value = pad_value, shuffle = True)

val_provider = DataProvider.Provider(stride = (4,4), input_size = input_size,
                           output_size = output_size, prediction_gap = prediction_gap,
                           batchsize = -1, pad = pad, pad_value = pad_value, shuffle = False)

test_provider = DataProvider.Provider(stride = stride, input_size = input_size,
                           output_size = output_size, prediction_gap = prediction_gap,
                           batchsize = -1, pad = pad, pad_value = pad_value, shuffle = False)

x = tf.placeholder(tf.float32, shape=[None, args.observations, args.input_x, args.input_y], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')



def stn(x, input_x, input_y, observations, reuse=False, name='stn'):

    with tf.variable_scope(name, reuse=reuse):
        network = tl.layers.InputLayer(x, name='input_layer')
        network = tl.layers.ReshapeLayer(network, shape = (-1, observations, input_x, input_y, 1))
        conv1 = tl.layers.Conv3dLayer(network, shape = [3, 3 ,3, 1, 3], strides=[1, 1, 1, 1, 1], name = 'conv1')
        lstm1 = LayerExtension.ConvRNNLayer(network, cell_shape = (input_x, input_y), cell_fn = ConvRNNCell.BasicConvLSTMCell, n_steps = observations, feature_map = 3,
                                             name='convlstm1')
        network = tl.layers.ConcatLayer([lstm1, conv1], concat_dim=4)
        conv2 = tl.layers.Conv3dLayer(network, shape = [3, 3 ,3, 6, 6], strides=[1, 1, 1, 1, 1], name = 'conv2')
        lstm2 = LayerExtension.ConvRNNLayer(network, cell_shape = (input_x, input_y), cell_fn = ConvRNNCell.BasicConvLSTMCell, n_steps = observations, feature_map = 6,
                                              name='convlstm2')
        network = tl.layers.ConcatLayer([lstm2, conv2], concat_dim=4, name = 'concat2')
        conv3 = tl.layers.Conv3dLayer(network, shape = [3, 3 ,3, 12, 12], strides=[1, 1, 1, 1, 1], name = 'conv3')
        lstm3 = LayerExtension.ConvRNNLayer(network, cell_shape = (input_x, input_y), cell_fn = ConvRNNCell.BasicConvLSTMCell, n_steps = observations, feature_map = 12,
                                              name='convlstm3')
        network = tl.layers.ConcatLayer([lstm3, conv3], concat_dim=4, name = 'concat3')
        network = tl.layers.FlattenLayer(network)
        network = tl.layers.DenseLayer(network, n_units=4096,
                                        act = tf.nn.relu ,
                                        name='dense1')
        network = tl.layers.DenseLayer(network, n_units=1024,
                                        act = tf.nn.relu,
                                        name='dense2')
        network = tl.layers.DenseLayer(network, n_units=1,
                                        act = tl.activation.identity,
                                        name='output_layer')
        return network


network = stn(x, args.input_x, args.input_y, args.observations)
y = network.outputs
cost = tl.cost.mean_squared_error(y, y_)
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables
sess.run(tf.initialize_all_variables())

if args.pre_model_path is not None:
    load_params = tl.files.load_npz(path=args.pre_model_path, name=args.pre_model_name)
    tl.files.assign_params(sess, load_params, network)

print 'set done'

nf.customfit(sess=sess, network=network, cost=cost, train_op=train_op, tra_provider=tra_provider, x=x, y_=y_, acc=None,
             n_epoch=args.epoch, print_freq=1, val_provider=val_provider, save_model=1, tra_kwag=tra_kwag, val_kwag=val_kwag,
             save_path=args.model_path + args.model_name, epoch_identifier=None)

if args.ouroboros_e > 0:
    nf.Ouroborosfit(sess=sess, network=network, cost = cost, train_op=train_op, x=x, y_=y_, dataset = tra_set,
                    batchsize=batchsize, input_size=input_size, pad = pad, n_epoch=args.ouroboros_e, mean=mean, std=std, val_provider=val_provider,
                    save_model=1, val_kwag=val_kwag ,save_path=args.model_path + args.model_name + '_ots', epoch_identifier=None)