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

from BoxEngine.BoxNetwork import BoxNetwork
from Utils import CheckpointLoader
import tensorflow as tf
from resnet_v2 import resnet_v2_101

slim = tf.contrib.slim

def fe_rpn(input):
    conv1 = slim.conv2d(input, 1024, [1, 3])
    conv2 = slim.conv2d(input, 1024, [3, 3])
    pool = slim.max_pool2d(input, [2, 2], stride=2)
    deconv = slim.convolution2d_transpose(pool, 1024, [2, 2], stride=2)
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
                 freezeBatchNorm=False,
                 reuse=False,
                 isTraining=True,
                 trainFrom=None,
                 hardMining=True):
        self.boxThreshold = 0.5

        if trainFrom == "0":
            trainFrom = "Conv2d_1a_3x3"
        elif trainFrom == "-1":
            trainFrom = None

        print("Training network from " + (trainFrom if trainFrom is not None else "end"))

        with tf.variable_scope(name, reuse=reuse) as scope:
            self.resNet, endpoints = resnet_v2_101(inputs,
                                                   is_training=isTraining,
                                                   scope=scope)
            res11 = endpoints[name + "/block3/unit_4/bottleneck_v2"]
            res16 = endpoints[name + "/block3/unit_9/bottleneck_v2"]
            res21 = endpoints[name + "/block3/unit_14/bottleneck_v2"]
            res27 = endpoints[name + "/block3/unit_20/bottleneck_v2"]
            res30 = endpoints[name + "/block3/unit_23/bottleneck_v2"]
            #res33 = endpoints[name + "/block4/unit_3/bottleneck_v2"]

            self.scope = scope

            with tf.variable_scope("Box"):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(weightDecay),
                                    biases_regularizer=slim.l2_regularizer(weightDecay),
                                    padding='SAME',
                                    activation_fn=tf.nn.relu):
                    rpnInput = fe_rpn(res30)
                    featureInput = hfg([res11, res16, res21, res27], 256)
                    # TODO: featureOffset, rpnOffset
                    BoxNetwork.__init__(self,
                                        nCategories,
                                        rpnInput,
                                        32,  # rpnDownscale
                                        [32, 32],  # rpnOffset
                                        featureInput,
                                        16,  # featureDownsample
                                        [16, 16],  # featureOffset
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
