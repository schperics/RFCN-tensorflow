# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf
import tensorflow.contrib.slim.python.slim.nets.resnet_v2 as resnet

from BoxEngine.BoxNetwork import BoxNetwork
from Utils import CheckpointLoader

slim = tf.contrib.slim


# slim의 논문에서의 resnet 구현은 convX_1 에서 size가 줄어들게 되어있다. 하지만 slim의 구현은 마지막 layer에서 줄어들게 되어있다.
# res30은 block3의 마지막 layer라서 원본의 1/32이고 block4는 stride가 1이라 크기가 줄지 않는다.
# 따라서 원래 resnet 논문대로 block의 첫 layer에서 사이즈가 줄어야 FEN에서 크가가 같은 layer끼리 모을수가 있다.
def resnet_v2_block(scope, base_depth, num_units, stride):
    return resnet.resnet_utils.Block(scope, resnet.bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }] + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1))


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
              resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
    return resnet.resnet_v2(inputs,
                            blocks,
                            num_classes,
                            is_training,
                            global_pool,
                            output_stride,
                            include_root_block=True,
                            reuse=reuse,
                            scope=scope)


def fe_rpn(input):
    conv1 = slim.conv2d(input, 1024, [1, 3])
    conv2 = slim.conv2d(input, 1024, [3, 3])
    pool = slim.max_pool2d(input, [2, 2], stride=2)
    deconv = slim.convolution2d_transpose(pool, 1024, [3, 3], stride=2)
    output = tf.concat((conv1, conv2, deconv), axis=-1)
    return output


def hfg(layers, depth):
    layers = [slim.conv2d(l, depth, [1, 1]) for l in layers]
    output = tf.concat(tuple(layers), axis=-1)
    return output


class BoxResnet(BoxNetwork):
    def __init__(self,
                 inputs,
                 nCategories,
                 name="BoxNetwork",
                 weightDecay=0.00004,
                 reuse=False,
                 isTraining=True,
                 hardMining=True):
        self.boxThreshold = 0.5

        with tf.variable_scope(name, reuse=reuse) as scope:
            with slim.arg_scope(resnet.resnet_arg_scope()):
                self.resNet, endpoints = resnet_v2_101(inputs,
                                                       is_training=isTraining,
                                                       global_pool=False,
                                                       scope=scope)
            res11 = endpoints[name + "/block3/unit_4/bottleneck_v2"]
            res16 = endpoints[name + "/block3/unit_9/bottleneck_v2"]
            res21 = endpoints[name + "/block3/unit_14/bottleneck_v2"]
            res27 = endpoints[name + "/block3/unit_20/bottleneck_v2"]
            res30 = endpoints[name + "/block3/unit_23/bottleneck_v2"]
            res33 = endpoints[name + "/block4/unit_3/bottleneck_v2"]

            self.scope = scope

            with tf.variable_scope("Box"):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(weightDecay),
                                    biases_regularizer=slim.l2_regularizer(weightDecay),
                                    padding='SAME',
                                    activation_fn=tf.nn.relu):
                    rpnInput = fe_rpn(res30)
                    featureInput = hfg([res11, res16, res21, res27, res33], 256)
                    BoxNetwork.__init__(self,
                                        nCategories,
                                        rpnInput,
                                        32,  # rpnDownscale
                                        [0, 0],  # rpnOffset
                                        featureInput,
                                        32,  # featureDownsample
                                        [0, 0],  # featureOffset
                                        weightDecay=weightDecay,
                                        hardMining=hardMining)

    # TODO : includeFeatures
    def getVariables(self, includeFeatures=False):
        if includeFeatures:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        else:
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name + "/Box/")
            vars += self.resNet.getTrainableVars()

            print("Training variables: ", [v.op.name for v in vars])
            return vars

    def importWeights(self, sess, filename):
        ignores = []
        """
        ignores = [] if includeTraining or (self.trainFrom is None) else self.getScopes(fromLayer=self.trainFrom,
                                                                                        inclusive=True)
        print("Ignoring blocks:")
        print(ignores)
        """
        CheckpointLoader.importIntoScope(sess,
                                         filename,
                                         fromScope="resnet_v2_101",
                                         toScope=self.scope.name,
                                         ignore=ignores)
