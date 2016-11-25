import numpy

class LogisticRegression:

    def __init__(self,data,classes, tolerancearg):
        self.tolerance = tolerancearg
        self.features = numpy.ones((data.shape[0],data.shape[1]+1))
        self.features[:,1:] = data
        self.labels = numpy.array(classes)
        self.labels = classes.reshape(classes.size,1)
        self.weights = numpy.zeros((data.shape[1]+1,1))
        ratio = int(0.8 * self.features.shape[0])
        self.trainfeatures = self.features[0:ratio]
        self.trainlabels = self.labels[0:ratio]
        self.testfeatures = self.features[ratio:]
        self.testlabels = self.labels[ratio:]
        self.shufflefeatures = self.trainfeatures
        self.shufflelabels = self.trainlabels
        self.lambdaa = 10
        print "size of training data is " + str(self.trainfeatures.shape[0])
        print "size of test data is " + str(self.testfeatures.shape[0])

    def calculateRowProbability(self,row):
        prob = 1 + numpy.exp(-self.shufflefeatures[row,:].dot(self.weights))
        return 1/prob

    def calculateProbability(self):
        prob = 1 + numpy.exp(-self.trainfeatures.dot(self.weights))
        return 1/prob

    def calculateLikelihoodGradient(self,row):
        temp = self.shufflelabels[row] - self.calculateRowProbability(row)
        ans = self.shufflefeatures[row,:] * temp
        return ans.reshape(self.features.shape[1],1)

    def calculateLikelihood(self,regularization):
        prob = self.calculateProbability()
        value = self.trainlabels * numpy.log(prob+1e-24) + (1-self.trainlabels) * numpy.log(1-prob+1e-24)
        if regularization == 1:
            return -1 * value.sum() + self.lambdaa * numpy.square(self.weights).sum(axis=0)
        else:
            return -1 * value.sum()

    def calculateStochasticGradientDescent(self,alpha,iterations,method,regularization):
        oldlikelihood = self.calculateLikelihood(regularization)
        diff = self.tolerance * 2
        i = 0
        velocity = numpy.zeros_like(self.weights)
        currentrows = range(len(self.trainfeatures))
        numpy.random.shuffle(currentrows)
        self.shufflefeatures = self.shufflefeatures[currentrows,:]
        self.shufflelabels = self.shufflelabels[currentrows]
        oldweights = self.weights
        oldvelocity = velocity
        mu = 0.60
        while diff > self.tolerance and i < iterations:
            if i%500 == 0:
                print "Iteration " + str(i)
            for j in range(len(self.trainfeatures)):
                if method == 1:
                    if regularization == 1:
                        self.weights = self.weights + alpha * self.calculateLikelihoodGradient(j) - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights + alpha * self.calculateLikelihoodGradient(j)
                elif method == 2:
                    velocity = mu * velocity + alpha * self.calculateLikelihoodGradient(j)
                    if regularization == 1:
                        self.weights = self.weights + velocity - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights + velocity
                elif method == 3:
                    velocity = mu * velocity + alpha * self.calculateLikelihoodGradient(j)
                    if regularization == 1:
                        self.weights = self.weights - mu * oldvelocity + (1 + mu) * velocity - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights - mu * oldvelocity + (1 + mu) * velocity
            newlikelihood = self.calculateLikelihood(regularization)
            diff = numpy.abs(newlikelihood - oldlikelihood)
            if newlikelihood > oldlikelihood:
                alpha = 0.5 * alpha
                self.weights = oldweights
            else:
                alpha = 1.05 * alpha
            oldweights = self.weights
            oldvelocity = velocity
            oldlikelihood = newlikelihood
            i = i + 1
            numpy.random.shuffle(currentrows)
            self.shufflefeatures = self.shufflefeatures[currentrows,:]
            self.shufflelabels = self.shufflelabels[currentrows]
        print self.weights.T
        print self.calculateLikelihood(regularization)
        a = raw_input("press enter")
        classified = 0
        count = 0
        for i in range(0,self.testfeatures.shape[0]):
            prob = 1 + numpy.exp(-self.testfeatures[i].dot(self.weights))
            label = 0.0
            if(1/prob) > 0.5:
                label = 1.0
                count = count + 1
            elif (1/prob) < 0.5:
                label = 0.0
                count = count + 1
            else:
                label = 1.0
                count = count + 1
            #print "label assigned is"+ str(label)
            if label != self.testlabels[i][0]:
                pass
            else:
                classified = classified + 1
        print 'Correctly classified test data is  ' + str(classified*100/count) + '%'

        classified = 0
        count = 0
        for i in range(0, self.trainfeatures.shape[0]):
            prob = 1 + numpy.exp(-self.trainfeatures[i].dot(self.weights))
            label = 0.0
            if (1 / prob) > 0.5:
                label = 1.0
                count = count + 1
            elif (1 / prob) < 0.5:
                label = 0.0
                count = count + 1
            else:
                label = 1.0
                count = count + 1
            if label != self.trainlabels[i][0]:
                pass
            else:
                classified = classified + 1
        print 'Correctly classified training data is  ' + str(classified * 100 / count) + '%'




