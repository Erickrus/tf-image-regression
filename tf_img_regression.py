import numpy as np
import tensorflow as tf
import os

from PIL import Image
from tf_img_regression_conf import TfImgRegressionConf

# This program is based on Andrej Karpathy's Image Regression Experiments
# Its original version is in javascript, now it is rewritten with python tensorflow
# For more information please refer to the below
# https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
# due to the random initalization, in some case the image will not be generated

class TfImgRegression():
    def __init__(self, imageFilename):
        self.imageFilename = imageFilename
        self.memoryFilename = imageFilename.replace(".jpg","-%3d.jpg").replace("img/","img/output/")
        os.system("mkdir img/output")
        self.layerId = 0
        self.args = TfImgRegressionConf()
        self._load_image(imageFilename)
    
    def sample(self):
        x = int(np.random.uniform(0, self.sourceImage.width-1))
        y = int(np.random.uniform(0, self.sourceImage.height-1))
        (r, g, b) = self.sourceImagePixels[x,y]
        return (
            np.array([float(x),
                      float(y)]),
            np.array([float(r)/float(255),
                      float(g)/float(255),
                      float(b)/float(255)]))
    
    def define_placeholders(self):
        with tf.name_scope(name="input"):
            self.xs = tf.placeholder(tf.float32, [None, 2], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, 3], name="ys")

    
    def _add_layer(self, inputs, inSize, outSize, actv=None):
        self.layerId += 1
        with tf.name_scope(name=("layer-%d" % self.layerId)):
            with tf.name_scope(name="weights"):
                Weights = tf.Variable(tf.random_normal([inSize, outSize]), name="W")
                
            with tf.name_scope(name="biases"):
                #recommend all biases not equal to 0.0
                biases = tf.add(tf.Variable(tf.zeros([1,outSize])), 0.1, name="b")
            
            with tf.name_scope(name="Wx_plus_b"):
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights),biases)
                
            if actv is None:
                outputs = Wx_plus_b
            else:
                outputs = actv(Wx_plus_b)
            
            return outputs

    def define_layers(self):
        currLayer = self._add_layer(self.xs, 2, self.args.neuronNum, actv=tf.nn.relu)
        for i in range(self.args.layerNum-2):
            currLayer = self._add_layer(currLayer, self.args.neuronNum, self.args.neuronNum, actv=tf.nn.relu)
        self.predictOp = self._add_layer(currLayer, self.args.neuronNum, 3, actv=tf.nn.relu)
    
    def define_optimizer(self):
        with tf.name_scope("loss"):
            self.lossOp = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(self.ys - self.predictOp),
                    reduction_indices=[1]
                )
            )
        
        with tf.name_scope("train"):
            # trying to use various optimizer based on the configuration, for the moment, eval is not used 
            if self.args.optimizerName == "GradientDescentOptimizer":
                self.trainOp = tf.train.GradientDescentOptimizer(self.args.learningRate).minimize(self.lossOp)
            if self.args.optimizerName == "RMSPropOptimizer":
                self.trainOp = tf.train.RMSPropOptimizer(self.args.learningRate).minimize(self.lossOp)
            if self.args.optimizerName == "AdamOptimizer":
                self.trainOp = tf.train.AdamOptimizer(self.args.learningRate).minimize(self.lossOp)
            print("using %s" % self.args.optimizerName)

        tf.summary.scalar("scalar_loss", self.lossOp)
        self.mergeSummaryOp = tf.summary.merge_all()
    
    def _adjust_color(self, color):
        return max(min(int(float(color) * 255.0),255),0)
        
    def train(self):
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.args.summaryPath, sess.graph)
            sess.run(tf.global_variables_initializer())
            epoch = 0
            
            for _ in range(self.args.epochNum):
                epoch += 1
                step = 0
                losses = []
                for _ in range(self.sourceImage.width * self.sourceImage.height):
                    step += 1
                    (position, color) = self.sample()
                    originalPosition = position.copy()
                    
                    position[0] = float(position[0]) / float(self.sourceImage.width)
                    position[1] = float(position[1]) / float(self.sourceImage.height)

                    position = position.reshape([1,2])
                    color = color.reshape([1,3])
                    summary, loss, _train, predict = sess.run([
                        self.mergeSummaryOp,
                        self.lossOp,
                        self.trainOp,
                        self.predictOp
                        ], 
                        feed_dict = {
                            self.xs: position,
                            self.ys: color
                        })
                    
                    if (len(losses)>=self.args.summaryPeriod):
                        losses = losses[1:]
                    losses.append(loss)
                    
                    # predict 1 pixel once
                    predictColor = (predict[0])
                    # draw the predict pixel
                    self.targetImagePixels[
                        originalPosition[0],
                        originalPosition[1]] = (
                        self._adjust_color(predictColor[0]),
                        self._adjust_color(predictColor[1]),
                        self._adjust_color(predictColor[2])
                    )
                    
                    train_writer.add_summary(summary)
                    if step % self.args.summaryPeriod == 0:
                        print( "epoch %d - step %d: %5.5f" % (epoch, step, np.mean(losses)))
                        print(predict)
                        print(color)
                        outputFilename = self.memoryFilename % (epoch)
                        outputFilename = outputFilename.replace(" ","0")
                        self.targetImage.save(outputFilename)

    def _load_image(self, filename):
        self.sourceImage = Image.open(filename).convert("RGB")
        self.sourceImagePixels = self.sourceImage.load()
        self.targetImage = Image.new("RGB", self.sourceImage.size, "black")
        self.targetImagePixels = self.targetImage.load()
        
    
    def main(self):
        self.define_placeholders()
        self.define_layers()
        self.define_optimizer()
        self.train()

if __name__ == "__main__":
    tim = TfImgRegression("./img/002.jpg")
    tim.main()