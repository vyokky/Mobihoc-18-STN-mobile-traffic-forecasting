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
    parser.add_argument('--model_path',
                        type=str,
                        default='./',
                        help='Saved model path')
    parser.add_argument('--model_name',
                        type=str,
                        default='stn',
                        help='Saved model name')
    parser.add_argument('--model_path_ots',
                        type=str,
                        default='./',
                        help='Saved model path of the OTS model')
    parser.add_argument('--model_name_ots',
                        type=str,
                        default='stn_ots',
                        help='Saved model name of the OTS model')
    parser.add_argument('--save_file',
                        type=str,
                        default='./prediction',
                        help='Saved prediction file')
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
    parser.add_argument('--empirical_mean',
                        type=str,
                        default='./data/',
                        help='this is the file of the empirical mean')
    parser.add_argument('--period_t',
                        type=int,
                        default=0,
                        help='the time index in the period for starting the prediction')
    parser.add_argument('--start_t',
                        type=int,
                        default=0,
                        help='the time index for starting the prediction')
    parser.add_argument('--step',
                        type=int,
                        default=60,
                        help='step of prediction')
    parser.add_argument('--fragment_size',
                        type=int,
                        default=1000,
                        help='batch size of prediction')
    parser.add_argument('--w',
                        type=float,
                        default=0.01,
                        help='parameter w in the OTS')
    parser.add_argument('--b',
                        type=float,
                        default=-5,
                        help='parameter b in the OTS')
    parser.add_argument('--delta',
                        type=float,
                        default=0.5,
                        help='parameter \delta in the OTS')
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
history = (np.load(args.empirical_mean) - args.mean) / args.std

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
frameshape = (test_set.shape[1], test_set.shape[2])

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
network_ots = stn(x, args.input_x, args.input_y, args.observations, name = 'ots_stn')
sess.run(tf.initialize_all_variables())
y = network.outputs

params  = tl.files.load_npz(path=args.model_path, name=args.model_name)
tl.files.assign_params(sess, params, network)
params_ots  = tl.files.load_npz(path=args.model_path_ots, name=args.model_name_ots)
tl.files.assign_params(sess, params_ots, network_ots)
# initialize all variables
print 'set done'


test_kwag = {
    'inputs': test_set[:,:,args.start_t-args.observations-1:args.start_t],
    'framebatch': framebatch,
    'mean': args.mean,
    'std': args.std,
    'norm_tar': True}

dstn_predictions = nf.out_futurepredictor(sess = sess, network=network, network2=network_ots, timestamp=args.period_t,
                                            season=history, output_provider = test_provider,  x = x, mean=args.mean,
                                            std=args.std, fragment_size=args.fragment_size, output_length=1, y_op=None,
                                            out_kwag=test_kwag, frameshape=frameshape, future=args.step, w=args.w, bias=args.b,
                                            low_weight=args.delta, weight_decay=1.0/args.observations)

prediction = dstn_predictions[0].reshape(-1,frameshape[0],frameshape[1])*args.std+args.mean
np.save(args.save_file, prediction)