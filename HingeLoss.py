import numpy

class HingeLoss:


#Initialize the class variables
#The variable names indicate what data is stored in them
    def __init__(self,data,classes, tolerancearg):
        self.tolerance = tolerancearg
        self.features = numpy.array(data)
        self.labels = numpy.array(classes)
        self.weights = numpy.array((0,0))
        ratio = int(0.8 * self.features.shape[0])
        self.trainfeatures = self.features[0:ratio]
        self.trainlabels = self.labels[0:ratio]
        self.testfeatures = self.features[ratio:]
        self.testlabels = self.labels[ratio:]
        self.shufflefeatures = self.trainfeatures
        self.shufflelabels = self.trainlabels
        self.lambdaa = 150

#Method to calculate the hinge loss value for the entire data set
    def hinge_loss(self,regularization):
            loss = 0
            for (x,y) in zip(self.trainfeatures,self.trainlabels):
                v = y*numpy.dot(self.weights,x)
                loss += max(0,1-v)
            if regularization == 1:
                return (loss  + self.lambdaa * numpy.square(self.weights).sum(axis=0))
            else:
                return (loss)
#Method to calculate the hinge loss value and gradient for a particular row (used for Stochastic Gradient)
    def hinge_loss_Stochastic(self,row):
        loss,grad = 0,0
        v = self.shufflelabels[row]*numpy.dot(self.weights,self.shufflefeatures[row])
        loss = max(0,1-v)
        grad = 0 if v > 1 else -self.shufflelabels[row]*self.shufflefeatures[row]
        return  grad


#Method to train classifier using Stochastic Gradient Method
    def calculateStochasticGradientDescent(self,alpha,iterations,method,regularization):
        oldloss = self.hinge_loss(regularization)
        diff = self.tolerance * 2
        i = 1
        velocity = numpy.zeros_like(self.weights)
        currentrows = range(len(self.trainfeatures))
        numpy.random.shuffle(currentrows)                           #Shuffle row numbers to make sure rows in the data set are random
        self.shufflefeatures = self.shufflefeatures[currentrows,:]
        self.shufflelabels = self.shufflelabels[currentrows]
        oldweights = self.weights
        oldvelocity = velocityk
        mu = 0.60                                                   #Parameter while calculating momentum
        file = open("HINGE.csv", "w")
        while diff > self.tolerance and i < iterations:
            file.write((str)(alpha)+"\n")
            if i%500 == 0:
                print "Iteration " + str(i)
            for j in range(len(self.trainfeatures)):

                #Method 1 is for training weights without momentum
                #Method 2 is for training weights with Classical momentum
                #Method 3 is for training weights with NAG 
                #Each one has been implemented with and without a regularization term
                if method == 1:
                    if regularization == 1:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j) - alpha *  self.lambdaa * self.weights
                    else:
                        self.weights = self.weights - alpha * self.hinge_loss_Stochastic(j)
                elif method == 2:
                    velocity = mu * velocity - alpha * self.hinge_loss_Stochastic(j)
                    if regularization == 1:
                        self.weights = self.weights + velocity - alpha *  self.lambdaa * self.weights
                    else:
                        self.weights = self.weights + velocity
                elif method == 3:
                    velocity = mu * velocity - alpha * self.hinge_loss_Stochastic(j)
                    if regularization == 1:
                        self.weights = self.weights - mu * oldvelocity + (1 + mu) * velocity - alpha * self.lambdaa * self.weights
                    else:
                        self.weights = self.weights - mu * oldvelocity + (1 + mu) * velocity
            newloss = self.hinge_loss(regularization)
            #Checking the difference between old likelihood and new likelihood and updating the learning rate
            diff = numpy.abs(newloss - oldloss)
            if newloss > oldloss:
                alpha = 0.5 * alpha                                         #If we overshoot reduce learning rate by 50%
                self.weights = oldweights
            else:
                alpha = 1.05 * alpha                                        #If we are going in the correct direction increase alpha by 5%
            oldweights = self.weights
            oldvelocity = velocity
            oldloss = newloss
            i = i + 1
            
            #Perform shuffling again before the next epoch
            numpy.random.shuffle(currentrows)
            self.shufflefeatures = self.shufflefeatures[currentrows,:]
            self.shufflelabels = self.shufflelabels[currentrows]
        print "The final weights are "+ str(self.weights.T)
        print "The final loss value is "+str(self.hinge_loss(regularization))
        print "Converged in " + str(i) + " iterations"
        file.close()
        classified = 0
        count = 0
        for i in range(0,self.testfeatures.shape[0]):
            prob = numpy.sign(numpy.dot(self.weights,self.testfeatures[i]))
            label = 0.0
            if(prob) == 1:
                label = 1.0
                count = count + 1
            elif (prob) == -1:
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
        print 'Correctly classified test data is  ' + str(classified*100.0/count) + '%'

        classified = 0
        count = 0
        for i in range(0, self.trainfeatures.shape[0]):
            prob = numpy.sign(numpy.dot(self.weights,self.trainfeatures[i]))
            label = 0.0
            if (prob) == 1:
                label = 1.0
                count = count + 1
            elif (prob) == -1:
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
        print 'Correctly classified training data is  ' + str(classified * 100.0 / count) + '%'




