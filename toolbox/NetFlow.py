import tensorlayer as tl
import tensorflow as tf
import numpy as np
import time
import copy
import DataProvider
import sys


def dict_to_one(dp_dict={}):

    """ Input a dictionary, return a dictionary that all items are
    set to one, use for disable dropout, drop-connect layer and so on.

    Parameters
    ----------
    dp_dict : dictionary keeping probabilities date
    """
    return {x: 1 for x in dp_dict}


def sigmoid(x):

    return 1/(1+np.exp(-x))


def modelsaver(network, path, epoch_identifier=None):

    if epoch_identifier:
        ifile = path + '_' + str(epoch_identifier)+'.npz'
    else:
        ifile = path + '.npz'
    tl.files.save_npz(network.all_params, name=ifile)


def customfit(sess, network, cost, train_op, tra_provider, x, y_, acc=None, n_epoch=50,
              print_freq=1, val_provider=None, save_model=-1, tra_kwag=None, val_kwag=None,
              save_path=None, epoch_identifier=None, baseline=10000000000000):
    """
        Train a given network by the given cost function, dataset, n_epoch etc.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        train_op : a TensorFlow optimizer
            like tf.train.AdamOptimizer
        x : placeholder
            for inputs
        y_ : placeholder
            for targets
        cost:  the TensorFlow expression of cost
        acc : the TensorFlow expression of accuracy (or other metric) or None
            if None, would not display the metric
        tra_provider :
            A object of DataProvider for training
        tra_kwag :
            Parameters dic. fed to the tra_provider
        val_provider :
            A object of DataProvider for validation
        val_kwag :
            Parameters dic. fed to the val_provider
        save_model :
            save model mode. 0 -- no save, -1 -- last epoch save, other positive int -- save frequency
        save_path :
            model save path
        epoch_identifier :
            save path + epoch? or not
        n_epoch : int
            the number of training epochs
        print_freq : int
            display the training information every ``print_freq`` epochs
        baseline: early stop first based line
    """

    # assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"
    print("Start training the network ...")

    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0;
        n_step = 0

        for batch in tra_provider.feed(**tra_kwag):
            X_train_a, y_train_a = batch
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep / n_step

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if val_provider is not None:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, train_acc, n_batch = 0, 0, 0
                for batch in tra_provider.feed(**tra_kwag):
                    X_train_a, y_train_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        train_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    train_loss += err;
                    n_batch += 1
                print("   train loss: %f" % (train_loss / n_batch))
                # print (train_loss, n_batch)
                if acc is not None:
                    print("   train acc: %f" % (train_acc / n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0

                for batch in val_provider.feed(**val_kwag):
                    X_val_a, y_val_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err;
                    n_batch += 1
                print("   val loss: %f" % (val_loss / n_batch))
                mean_val_loss = val_loss / n_batch
                if acc is not None:
                    print("   val acc: %f" % (val_acc / n_batch))
            else:
                print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))

        # print(save_model > 0, epoch % save_model == 0, epoch/save_model > 0)

        if save_model > 0 and epoch % save_model == 0:
            if epoch_identifier:
                modelsaver(network=network, path=save_path, epoch_identifier=epoch)
            else:
                modelsaver(network=network, path=save_path, epoch_identifier=None)

        elif save_model == -100:
            if mean_val_loss < baseline:
                modelsaver(network=network, path=save_path, epoch_identifier=None)
                baseline = mean_val_loss

    if save_model == -1:
        modelsaver(network=network, path=save_path, epoch_identifier=None)

    print("Total training time: %fs" % (time.time() - start_time_begin))


def Ouroborosfit(sess, network, cost, dataset, train_op, batchsize, input_size, x, y_, pad, n_epoch=50,
                 val_provider=None, save_model=-1, val_kwag=None, save_path=None, epoch_identifier=None, mean=0, std=1,
                 shuffle=True, print_frame_loss=True):
    """

    :param sess: TensorFlow session
            sess = tf.InteractiveSession()
    :param network: a TensorLayer layer
            the network will be trained
    :param cost: cost function
    :param dataset: raw dataset
    :param train_op: training optimiser
    :param batchsize: batch size
    :param input_size: network input size
    :param x: placeholder input
    :param y_: placeholder output
    :param pad: pad of input
    :param n_epoch: number of epoch
    :param val_provider: DataProvider for validation
    :param save_model: save model mode
    :param val_kwag: parameters dic. fed to the val_provider
    :param save_path: model save path
    :param epoch_identifier: path + epoch? or not
    :param mean: normalised constant mean
    :param std: normalised constant std
    :param shuffle: shuffle data or not
    :param print_frame_loss: print per frame loss or not
    :return: None
    """

    for epoch in range(n_epoch):
        start_time = time.time()
        for frame in xrange(dataset.shape[-1]-input_size[-1]):
            output_provider = DataProvider.Provider(stride=(1, 1), input_size=input_size,
                                                    output_size=(1, 1, 1), prediction_gap=0, batchsize=-1, pad=pad,
                                                    pad_value=0, shuffle=False)
            if frame == 0:
                input_source = dataset[:, :, :input_size[-1]]
            else:
                out_kwag = {
                    'inputs': input_source,
                    'framebatch': 1,
                    'mean': mean,
                    'std': std,
                    'norm_tar': True}

                frame_prediction = custompredict(sess=sess, network=network, output_provider=output_provider, x=x,
                                                 fragment_size=1000, output_length=1, y_op=None, out_kwag=out_kwag)
                frame_prediction = frame_prediction[0].reshape(dataset.shape[0], dataset.shape[1], 1)*std+mean

                input_source = np.concatenate([input_source[:, :, 1:], frame_prediction], axis=2)

            net_input, = output_provider.feed(inputs=input_source, framebatch=1, mean=mean, std=std, norm_tar=True)
            tra_provider = DataProvider.DoubleSourceProvider(batchsize=batchsize, shuffle=shuffle)
            ground_truth = dataset[:, :, input_size[-1] + frame].reshape(-1, 1)

            tra_kwag = {
                'inputs': net_input[0],
                'targets': (ground_truth-mean)/std
            }

            print 'prediction:', np.mean(np.mean(input_source, axis=0), axis=0)[-1], 'GT:', dataset[:, :, input_size[-1] + frame-1].mean()

            if print_frame_loss:
                print ("Epoch %d, frame %d of %d" % (epoch + 1, frame, dataset.shape[-1]-input_size[-1])),
            easyfit(sess=sess, network=network, cost=cost, train_op=train_op, tra_provider=tra_provider, x=x,
                    y_=y_, n_epoch=1, tra_kwag=tra_kwag, print_loss=print_frame_loss)
            sys.stdout.flush()

        if val_provider is not None:
            customtest(sess=sess, network=network, acc=None, test_provider=val_provider, x=x, y_=y_, cost=cost,
                       test_kwag=val_kwag)

        if save_model > 0 and epoch % save_model == 0:
            if epoch_identifier:
                modelsaver(network=network, path=save_path, epoch_identifier=epoch)
            else:
                modelsaver(network=network, path=save_path, epoch_identifier=None)

        print 'Epoch took:', time.time()-start_time, 's'

    if save_model == -1:
        modelsaver(network=network, path=save_path, epoch_identifier=None)


def custompredict(sess, network, output_provider, x, fragment_size=1000, output_length=1, y_op=None, out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
            the output
        output_provider : DataProvider
        out_kwag :
            Parameter dic. fed to the output_provider
        fragment_size :
            data number predicted for every step
        output_length :
            output size

    """
    dp_dict = dict_to_one(network.all_drop)  # disable noise layers

    if y_op is None:
        y_op = network.outputs
    output_container = []
    gt = []
    banum = 0
    for batch in output_provider.feed(**out_kwag):
        # print banum
        banum += 1
        X_out_a, gt_batch = batch
        # print 'hi', X_out_a.mean()
        fra_num = X_out_a.shape[0] / fragment_size
        offset = X_out_a.shape[0] % fragment_size
        final_output = np.zeros((X_out_a.shape[0], output_length))
        for fragment in xrange(fra_num):
            x_fra = X_out_a[fragment * fragment_size:(fragment + 1) * fragment_size]
            feed_dict = {x: x_fra, }
            feed_dict.update(dp_dict)
            final_output[fragment * fragment_size:(fragment + 1) * fragment_size] = \
                sess.run(y_op, feed_dict=feed_dict).reshape(-1, output_length)

        if offset > 0:
            feed_dict = {x: X_out_a[-offset:], }
            feed_dict.update(dp_dict)
            final_output[-offset:] = sess.run(y_op, feed_dict=feed_dict).reshape(-1, output_length)
        output_container.append(final_output)
        gt.append(gt_batch)
        # print 'hello', final_output.mean()
    return np.vstack(output_container), np.vstack(gt)


def out_futurepredictor(sess, network, network2, output_provider, x, timestamp, season, mean=0, std=1, fragment_size=1000,
                        output_length=1, y_op=None, out_kwag=None, frameshape=(100, 100), future=None, w=0, bias=0,
                        low_weight=0.5, weight_decay=1):
    """

    :param sess: TensorFlow session
            sess = tf.InteractiveSession()
    :param network: a TensorLayer layer
            the network will be used for prediction
    :param network2: the second TensorLayer layer
            the network will be used for prediction (mixed with the first one)
    :param output_provider: DataProvider
    :param x: placeholder
            the input
    :param mean: normalised constant
    :param std: normalised constant
    :param fragment_size: predictions made per steps, to fit in the memory
    :param output_length: predictions size
    :param y_op: the output operation
    :param out_kwag: parameters dic. fed in the output_provider
    :param frameshape: raw data frame shape
    :param future: how long is the prediction?
    :param weight: weight of two model
    :return: prediction, and groundtruth
    :param w: sigmoid weight w
    :param bias: sigmoid bias w
    :param low_weight: low bound of weight with two models
    :param timestamp: current time stamp of the empirical mean
    :param season: empirical mean
    :param weight_decay: change rate of weight with empirical mean
    :return:
    """

    if future is None:
        return custompredict(sess=sess, network=network, output_provider=output_provider,
                             x=x, fragment_size=fragment_size, output_length=output_length,
                             y_op=y_op, out_kwag=out_kwag)

    else:
        assert type(future) == int
        period = season.shape[-1]
        framesize = frameshape[0] * frameshape[1]
        truth_prediction, groundtruth = custompredict(sess=sess, network=network, output_provider=output_provider,
                             x=x, fragment_size=fragment_size, output_length=output_length,
                             y_op=y_op, out_kwag=out_kwag)
        whole_prediction = np.zeros((truth_prediction.size + future*framesize, 1))
        whole_prediction[0:truth_prediction.size] = truth_prediction
        input_size = (frameshape[0], frameshape[1], output_provider.input_size[2])
        print 'current done.'
        for frame in xrange(future):

            decay = 1 - sigmoid(w * frame + bias)
            print 'predicting the future:', frame, 'decay:', decay

            if frame == 0:
                whole_prediction[truth_prediction.size - framesize:truth_prediction.size] = \
                    decay * whole_prediction[truth_prediction.size - framesize:truth_prediction.size] +\
                    (1 - decay) * season[:, :, (timestamp + frame) % period].reshape(framesize, 1)

            future_input = np.zeros(input_size)
            if frame < (output_provider.input_size[2]-1):
                future_input[:, :, 0:output_provider.input_size[2]-frame-1]=\
                    out_kwag['inputs'][:, :, -(output_provider.input_size[2]-frame-1):]

                future_input[:, :, -(frame + 1):] = np.transpose((whole_prediction
                                            [truth_prediction.size-framesize:
                                            truth_prediction.size+(frame)*framesize]*std+mean).reshape(
                    -1, frameshape[0], frameshape[1]), (1, 2, 0))

            else:
                future_input = np.transpose((whole_prediction
                                            [truth_prediction.size+(frame-output_provider.input_size[2])*framesize:
                                            truth_prediction.size+(frame)*framesize]*std+mean).reshape(
                    -1, frameshape[0], frameshape[1]), (1, 2, 0))

            future_input = np.clip(future_input, 0, np.inf)
            print np.mean(np.mean(future_input, axis=0), axis=0)


            future_kwag = copy.deepcopy(out_kwag)
            future_kwag['inputs'] = future_input
            output_provider.prediction_gap = 0

            future_prediction, fake_gt = custompredict(sess=sess, network=network, output_provider=output_provider,
                          x=x, fragment_size=fragment_size, output_length=output_length, y_op=y_op,
                          out_kwag=future_kwag)

            if network2 is not None:
                weight = np.max([1 - weight_decay*frame, low_weight])
                future_prediction2, fake_gt2 = custompredict(sess=sess, network=network2, output_provider=output_provider,
                                                       x=x, fragment_size=fragment_size, output_length=output_length,
                                                       y_op=y_op, out_kwag=future_kwag)

                whole_prediction[truth_prediction.size + frame * framesize:truth_prediction.size + (frame + 1) *
                    framesize] = decay*(future_prediction*weight + future_prediction2*(1-weight)) + \
                                 (1-decay)*season[:, :, (timestamp+frame) % period].reshape(framesize, 1)
                print 'Difference (origin-ouroboros):', (future_prediction - future_prediction2).mean() * 1000

            else:
                whole_prediction[truth_prediction.size + frame * framesize:truth_prediction.size +
                                                                           (frame + 1) * framesize] = \
                    decay * future_prediction + (1-decay) * season[:, :, (timestamp+frame) % period].reshape(framesize, 1)

        return whole_prediction, groundtruth


def futurepredictor(sess, network, network2, output_provider, x, mean=0, std=1, fragment_size=1000, output_length=1,
                    y_op=None, out_kwag=None, frameshape=(100, 100), future=None, weight=0.5):

    """

    :param sess: TensorFlow session
            sess = tf.InteractiveSession()
    :param network: a TensorLayer layer
            the network will be used for prediction
    :param network2: the second TensorLayer layer
            the network will be used for prediction (mixed with the first one)
    :param output_provider: DataProvider
    :param x: placeholder
            the input
    :param mean: normalised constant
    :param std: normalised constant
    :param fragment_size: predictions made per steps, to fit in the memory
    :param output_length: predictions size
    :param y_op: the output operation
    :param out_kwag: parameters dic. fed in the output_provider
    :param frameshape: raw data frame shape
    :param future: how long is the prediction?
    :param weight: weight of two model
    :return: prediction, and groundtruth
    """

    if future is None:
        return custompredict(sess=sess, network=network, output_provider=output_provider,
                             x=x, fragment_size=fragment_size, output_length=output_length,
                             y_op=y_op, out_kwag=out_kwag)

    else:
        assert type(future) == int
        framesize = frameshape[0] * frameshape[1]
        truth_prediction, groundtruth = custompredict(sess=sess, network=network, output_provider=output_provider,
                             x=x, fragment_size=fragment_size, output_length=output_length,
                             y_op=y_op, out_kwag=out_kwag)
        whole_prediction = np.zeros((truth_prediction.size + future*framesize, 1))
        whole_prediction[0:truth_prediction.size] = truth_prediction
        input_size = (frameshape[0], frameshape[1], output_provider.input_size[2])
        print 'current done.'
        for frame in xrange(future):
            print 'predicting the future:', frame
            future_input = np.zeros(input_size)
            if frame < (output_provider.input_size[2]-1):
                future_input[:, :, 0:output_provider.input_size[2]-frame-1]=\
                    out_kwag['inputs'][:, :, -(output_provider.input_size[2]-frame-1):]

                future_input[:, :, -(frame + 1):] = np.transpose((whole_prediction
                                            [truth_prediction.size-framesize:
                                            truth_prediction.size+(frame)*framesize]*std+mean).reshape(
                    -1, frameshape[0], frameshape[1]), (1, 2, 0))#*(1+np.exp(-(frame+1)*input_decay)+bias)

            else:
                future_input = np.transpose((whole_prediction
                                            [truth_prediction.size+(frame-output_provider.input_size[2])*framesize:
                                            truth_prediction.size+(frame)*framesize]*std+mean).reshape(
                    -1, frameshape[0], frameshape[1]), (1, 2, 0))#*(1+np.exp(-(frame+1)*input_decay)+bias)

            # future_input = np.clip(future_input, 0, 8100)
            print np.mean(np.mean(future_input, axis=0), axis=0)

            future_kwag = copy.deepcopy(out_kwag)
            future_kwag['inputs'] = future_input
            output_provider.prediction_gap = 0

            future_prediction, fake_gt = custompredict(sess=sess, network=network, output_provider=output_provider,
                          x=x, fragment_size=fragment_size, output_length=output_length, y_op=y_op,
                          out_kwag=future_kwag)

            if network2 is not None:

                future_prediction2, fake_gt2 = custompredict(sess=sess, network=network2, output_provider=output_provider,
                                                       x=x, fragment_size=fragment_size, output_length=output_length,
                                                       y_op=y_op, out_kwag=future_kwag)

                whole_prediction[truth_prediction.size + frame * framesize:truth_prediction.size +
                                                                           (frame + 1) * framesize] = \
                    (future_prediction*weight + future_prediction2*(1-weight))
                print 'Difference (origin-ouroboros):', (future_prediction - future_prediction2).mean() * 1000

            else:
                whole_prediction[truth_prediction.size + frame * framesize:truth_prediction.size +
                                                                           (frame + 1) * framesize] = future_prediction
        return whole_prediction, groundtruth