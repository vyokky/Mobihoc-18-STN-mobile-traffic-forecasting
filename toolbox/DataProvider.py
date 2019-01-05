import numpy as np
import math


class Provider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values

f
    """

    def __init__(self, input_size, output_size, prediction_gap, flatten=True, batchsize=-1, stride=(1, 1),
                 shuffle=True, pad=None, pad_value=0):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.prediction_gap = prediction_gap
        self.stride = stride
        self.shuffle = shuffle
        self.pad = pad
        self.pad_value = pad_value
        self.flatten = flatten

    def DataSlicer_3D(self, inputs, excerpt, flatten=False, external=None):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, used for LOP blended.

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        if self.pad is None:
            pad_x = 0
            pad_y = 0
        else:
            pad_x = self.pad[0]
            pad_y = self.pad[1]

        x_max, y_max, z_max = inputs.shape
        x_max += pad_x * 2
        y_max += pad_y * 2
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], self.output_size[2], total))
        if external:
            external_data = np.zeros((total, 1))

        x_offset = (self.input_size[0] - self.output_size[0]) / 2
        y_offset = (self.input_size[1] - self.output_size[1]) / 2

        data_num = 0

        for frame in xrange(len(excerpt)):

            if external:
                external_frame = inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap]

            if self.pad is None:
                input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
                target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap - 1:
                excerpt[frame] + self.input_size[2] + self.prediction_gap + self.output_size[2] - 1].reshape(
                    self.output_size)

            else:
                input_frame = np.ones((x_max, y_max, self.input_size[2])) * self.pad_value
                target_frame = np.ones((x_max, y_max, self.output_size[2])) * self.pad_value

                input_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]

                target_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap - 1: excerpt[frame] \
                                                                                                + self.input_size[
                                                                                                    2] + self.prediction_gap +
                                                                                                self.output_size[2] - 1]

                target_frame = target_frame.reshape(x_max, y_max, self.output_size[2])

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, :, data_num] = target_frame[
                                                     x - self.input_size[0] + x_offset:x - self.input_size[0] +
                                                                                       x_offset + self.output_size[0],
                                                     y - self.input_size[1] + y_offset:y - self.input_size[1] +
                                                                                       y_offset + self.output_size[1],
                                                     :]
                    if external:
                        external_data[data_num] = external_frame[x - self.input_size[0] + x_offset,
                                                                 y - self.input_size[1] + y_offset]

                    data_num += 1

        if external:
            if self.shuffle:
                indices = np.random.permutation(total)
                if flatten:
                    return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            external_data[indices[0:self.batchsize]])
                return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        external_data[indices[0:self.batchsize]])

            else:
                if flatten:
                    return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                            np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)).flatten(),
                            external_data[0:self.batchsize])
                return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                        np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)),
                        external_data[0:self.batchsize])
        else:
            if self.shuffle:
                indices = np.random.permutation(total)
                if flatten:
                    return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))
                return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))

            else:
                if flatten:
                    return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)).flatten())
                return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] - self.prediction_gap - self.output_size[2] + 2

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, flatten=self.flatten)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std)).reshape(self.batchsize, -1)

            yield (net_inputs - mean) / float(std), net_targets

class DoubleSourceProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, batchsize, shuffle):

        self.batchsize = batchsize
        self.shuffle = shuffle

    def feed(self, inputs, targets):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        assert len(inputs) == len(targets)
        if self.batchsize < 0:
            self.batchsize = len(inputs)
        if self.shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]