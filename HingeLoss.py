import numpy

class HingeLoss:

    def __init__(self,data,classes, tolerancearg):
        self.tolerance = tolerancearg
        #self.features = numpy.ones((data.shape[0],data.shape[1]+1))
        self.features = numpy.array(data)
        self.labels = numpy.array(classes)
        #print self.labels
        #a = raw_input("Press enter")
        #print self.features
        #a = raw_input("Press enter")
        self.weights = numpy.array((0,0))
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

    def hinge_loss(self,regularization):
            loss,grad = 0,0
            for (x_,y_) in zip(self.trainfeatures,self.trainlabels):
                v = y_*numpy.dot(self.weights,x_)
                loss += max(0,1-v)
                grad += 0 if v > 1 else -y_*x_
            if regularization == 1:
                return (loss + self.lambdaa * numpy.square(self.weights).sum(axis=0),grad)
            else:
                return (loss,grad)

    def hinge_loss_Stochastic(self,row):
        """ evaluates hinge loss and its gradient at w

        rows of x are data points
        y is a vector of labels
        """
        loss,grad = 0,0
        v = self.shufflelabels[row]*numpy.dot(self.weights,self.shufflefeatures[row])
        loss += max(0,1-v)
        grad += 0 if v > 1 else -self.shufflelabels[row]*self.shufflefeatures[row]
        return  grad


    def calculateStochasticGradientDescent(self,alpha,iterations,method,regularization):
        oldloss = self.hinge_loss(regularization)[0]
        diff = self.tolerance * 2
        i = 1
        velocity = numpy.zeros_like(self.weights)
        currentrows = range(len(self.trainfeatures))
        numpy.random.shuffle(currentrows)
        self.shufflefeatures = self.shufflefeatures[currentrows,:]
        self.shufflelabels = self.shufflelabels[currentrows]
        oldweights = self.weights
        oldvelocity = velocity
        mu = 0.60
        file = open("HINGEwithL2.csv", "w")
        while diff > self.tolerance and i < iterations:
            file.write((str)(oldloss)+"\n")
            if i%500 == 0:
                print "Iteration " + str(i)
            for j in range(len(self.trainfeatures)):
                if method == 1:
                    if regularization == 1:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j) - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j)
                elif method == 2:
                    velocity = mu * velocity - alpha * self.hinge_loss_Stochastic(j)
                    if regularization == 1:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j) - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights + velocity
                elif method == 3:
                    velocity = mu * velocity - alpha * self.hinge_loss_Stochastic(j)
                    if regularization == 1:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j) - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights - mu * oldvelocity + (1 + mu) * velocity
            newloss = self.hinge_loss(regularization)[0]
            diff = numpy.abs(newloss - oldloss)
            #a = raw_input("Press enter")
            if newloss > oldloss:
                alpha = 0.5 * alpha
                self.weights = oldweights
            else:
                alpha = 0.80 * alpha
            oldweights = self.weights
            oldvelocity = velocity
            oldloss = newloss
            i = i + 1
            numpy.random.shuffle(currentrows)
            self.shufflefeatures = self.shufflefeatures[currentrows,:]
            self.shufflelabels = self.shufflelabels[currentrows]
            #print newloss
            #print alpha
        print self.weights
        print self.hinge_loss(regularization)[0]
        print "Converged in " + str(i) + "iterations"
        a = raw_input("press enter")
        classified = 0
        count = 0
        for i in range(0,self.testfeatures.shape[0]):
            prob = 1 + numpy.exp(-self.testfeatures[i].dot(self.weights))
            #print "probability is " + str(1 / prob)
            #print "the class should be " + str(self.testlabels[i][0])
            label = 0.0
            if(1/prob) > 0.5:
                label = 1.0
                count = count + 1
            elif (1/prob) < 0.5:
                label = -1.0
                count = count + 1
            else:
                label = 1.0
                count = count + 1
            #print "label assigned is"+ str(label)
            if label != self.testlabels[i]:
                pass
            else:
                classified = classified + 1
        print 'Correctly classified test data is  ' + str(classified*100/count) + '%'

        classified = 0
        count = 0
        for i in range(0, self.trainfeatures.shape[0]):
            prob = 1 + numpy.exp(-self.trainfeatures[i].dot(self.weights))
            #print "probability is " + str(1 / prob)
            #print "the class should be " + str(self.trainlabels[i][0])
            label = 0.0
            if (1 / prob) > 0.5:
                label = 1.0
                count = count + 1
            elif (1 / prob) < 0.5:
                label = -1.0
                count = count + 1
            else:
                label = 1.0
                count = count + 1
            #print "label assigned is" + str(label)
            if label != self.trainlabels[i]:
                pass
            else:
                classified = classified + 1
        print 'Correctly classified training data is  ' + str(classified * 100 / count) + '%'




