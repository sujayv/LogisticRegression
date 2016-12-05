import numpy

class LogisticRegression:

#Initialize the variables for the class
#The variable names indicate what data is stored in them
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
        self.lambdaa = 150


#Calculate the probability using sigmoid function for every row (to feed to calculateLikelihoodGradient)
    def calculateRowProbability(self,row):
        prob = 1 + numpy.exp(-(self.shufflefeatures[row,:].dot(self.weights)))
        return 1/prob


#Calculate probability using sigmoid function for entire data
    def calculateProbability(self):
        prob = 1 + numpy.exp(-(self.trainfeatures.dot(self.weights)))
        return 1/prob


#Calculate the gradient of the negative log likelihood and return
    def calculateLikelihoodGradient(self,row):
        temp = self.shufflelabels[row] - self.calculateRowProbability(row)
        ans = self.shufflefeatures[row,:] * temp * self.calculateRowProbability(row) * (1- self.calculateRowProbability(row))
        return ans.reshape(self.features.shape[1],1)


#Calculate the likelihood of the data
    def calculateLikelihood(self,regularization):
        prob = self.calculateProbability()
        value = self.trainlabels * numpy.log(prob+1e-24) + (1-self.trainlabels) * numpy.log(1-prob+1e-24)
        if regularization == 1:
            return -1 * value.sum() + self.lambdaa * numpy.square(self.weights).sum(axis=0)			#The second part is the L2 regularization term
        else:
            return -1 * value.sum()


#The main method to train our classifier using Stochastic Gradient Descent
    def calculateStochasticGradientDescent(self,alpha,iterations,method,regularization):
        oldlikelihood = self.calculateLikelihood(regularization)
        diff = self.tolerance * 2
        i = 0																		#Iteration count
        velocity = numpy.zeros_like(self.weights)									#Initial value of velocity
        currentrows = range(len(self.trainfeatures))
        numpy.random.shuffle(currentrows)											#Shuffle row numbers to make sure rows in the data set are random
        self.shufflefeatures = self.shufflefeatures[currentrows,:]
        self.shufflelabels = self.shufflelabels[currentrows]
        oldweights = self.weights
        oldvelocity = velocity
        mu = 0.60																	#Parameter while calculating momentum
        file = open("LOG.csv","w")
        while diff > self.tolerance and i < iterations:								#The loop to go through multiple epochs until tolerance is reached
            file.write((str)(alpha)+"\n")											#Output data to a .csv file
            if i%500 == 0:
                print "Epoch " + str(i)
                print oldlikelihood
            for j in range(len(self.trainfeatures)):

            	#Method 1 is for training weights without momentum
            	#Method 2 is for training weights with Classical momentum
            	#Method 3 is for training weights with NAG 
            	#Each one has been implemented with and without a regularization term
                if method == 1:
                    if regularization == 1:
                        self.weights = self.weights + alpha * self.calculateLikelihoodGradient(j) - alpha * self.lambdaa * self.weights			#Updating weights with regularization
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
            #Checking the difference between old likelihood and new likelihood and updating the learning rate
            if newlikelihood > oldlikelihood:				#If we overshoot reduce learning rate by 50%
                alpha = 0.50 * alpha
                self.weights = oldweights
            else:
                alpha = 1.05 * alpha						#If we are going in the correct direction increase alpha by 5%
            oldweights = self.weights
            oldvelocity = velocity
            oldlikelihood = newlikelihood
            i = i + 1
            
            #Perform shuffling again before the next epoch
            numpy.random.shuffle(currentrows)			
            self.shufflefeatures = self.shufflefeatures[currentrows,:]
            self.shufflelabels = self.shufflelabels[currentrows]

            
        print "The final weights are "+ str(self.weights.T)
        print "The final likelihood value is "+ str(self.calculateLikelihood(regularization))
        print "Converged in "+ str(i) + " iterations"
        file.close()
        #a = raw_input("press enter")
        classified = 0
        count = 0
        #Using the sigmoid function to check the classification of the test data
        for i in range(0,self.testfeatures.shape[0]):
            prob = 1 + numpy.exp(-(self.testfeatures[i].dot(self.weights)))
            label = 0.0
            if(1/prob) > 0.5:
                label = 1.0
                count = count + 1
            elif (1/prob) < 0.5:
                label = 0.0
                count = count + 1
            else:
                label = 1.0							#Since probability is 0.5 we can randomly classify it to any category
                count = count + 1
            #print "label assigned is"+ str(label)
            if label != self.testlabels[i][0]:
                pass
            else:
                classified = classified + 1
        print 'Correctly classified test data is  ' + str(classified*100.0/count) + '%'

        #Using sigmoid function to check the classification of the training data
        classified = 0
        count = 0
        for i in range(0, self.trainfeatures.shape[0]):
            prob = 1 + numpy.exp(-(self.trainfeatures[i].dot(self.weights)))
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
        print 'Correctly classified training data is  ' + str(classified * 100.0 / count) + '%'




