
class TfImgRegressionConf:
    def __init__(self):
        self.learningRate = 0.01
        self.initialBias = 0.0
        self.epochNum = 100
        self.layerNum = 3
        self.neuronNum = 20
        self.summaryPeriod = 10000
        self.summaryPath = "logs"
        self.optimizerName = "RMSPropOptimizer"
        # self.optimizerName = "GradientDescentOptimizer"
        # self.optimizerName = "AdamOptimizer"