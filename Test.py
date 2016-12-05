from LogisticRegression import LogisticRegression as LR
import numpy
import matplotlib.pyplot as plt
from HingeLoss import HingeLoss as HL

staticchoice = raw_input("Please enter your choice\n1. Use the same data set used to create the results of our report\n2. Create a new data set for classification");

#X is the numpy array which will contain the feature vectors
X = None
#y is the numpy array which will contain the feature labels(0 or 1)
y = None
#y_hl is the numpy array which will contain the feature labels for the hinge loss(-1 or 1)
y_hl = None



#Please refer to 'else' part of statements for what value each variable is storing
if(int(staticchoice) == 1):
	x = 8
	y = 8
	mean1 = numpy.array([int(x),int(y)])
	x1 = 3
	y1 = 0
	x2 = 0
	y2 = 3
	covariance1 = numpy.array([[int(x1),int(y1)],[int(x2),int(y2)]])
	n1 = 500
	X1 = numpy.random.multivariate_normal(mean1,covariance1,int(n1))

	x = 11
	y = 11
	mean2 = numpy.array([int(x),int(y)])
	x1 = 1
	y1 = 0
	x2 = 0
	y2 = 8
	covariance2 = numpy.array([[int(x1),int(y1)],[int(x2),int(y2)]])
	n2 = 600
	X2 = numpy.random.multivariate_normal(mean2,covariance2,int(n2))


	X = numpy.concatenate((X1,X2),axis=0)
	y = numpy.array([0]*int(n1) + [1]*int(n2))
	y_hl = numpy.array([-1]*int(n1) + [1]*int(n2))
else:
	x = raw_input("enter x co-ordinate of mean1 ")
	y = raw_input("enter y co-ordinate of mean1 ")
	mean1 = numpy.array([int(x),int(y)])										#Generate the mean numpy array
	x1 = raw_input("enter x1 of the 2-dimensional covariance matrix ")
	y1 = raw_input("enter y1 of the 2-dimensional covariance matrix ")
	x2 = raw_input("enter x2 of the 2-dimensional covariance matrix ")
	y2 = raw_input("enter y2 of the 2-dimensional covariance matrix ")
	covariance1 = numpy.array([[int(x1),int(y1)],[int(x2),int(y2)]])			#Generate the covariance numpy array
	n1 = raw_input("enter the number of samples of first class ")
	X1 = numpy.random.multivariate_normal(mean1,covariance1,int(n1))			#Numpy method to generate correlated data

	x = aw_input("enter x co-ordinate of mean2 ")
	y = raw_input("enter y co-ordinate of mean2 ")
	mean2 = numpy.array([int(x),int(y)])
	x1 = raw_input("enter x1 of the 2-dimensional covariance matrix ")
	y1 = raw_input("enter y1 of the 2-dimensional covariance matrix ")
	x2 = raw_input("enter x2 of the 2-dimensional covariance matrix ")
	y2 = raw_input("enter y2 of the 2-dimensional covariance matrix ")
	covariance2 = numpy.array([[int(x1),int(y1)],[int(x2),int(y2)]])
	n2 = raw_input("enter the number of samples of second class ")
	X2 = numpy.random.multivariate_normal(mean2,covariance2,int(n2))


	X = numpy.concatenate((X1,X2),axis=0)
	y = numpy.array([0]*int(n1) + [1]*int(n2))
	y_hl = numpy.array([-1]*int(n1) + [1]*int(n2))


#Plot the figure
fig = plt.figure()


#Shuffle the rows so that data with same labels are not one after the other as we are concatenating two arrays earlier.
currrows = range(len(X))
numpy.random.shuffle(currrows)
numpy.random.shuffle(currrows)
numpy.random.shuffle(currrows)
X = X[currrows,:]
y = y[currrows]
y_hl = y_hl[currrows]


#Block to standardize the data
X_mean = X.mean(axis=0)							#Calculate mean of feature vectors
X_deviation = X.std(axis=0)						#Calculate the standard deviation

X = (X - X_mean) / X_deviation;					#Z score normalization formula

plt.scatter(X1[:,0],X1[:,1], marker='+')				#Plot the generated data on matplotlib
plt.scatter(X2[:,0],X2[:,1], c= 'green', marker='o')	#Plot the generated data on maptplotlib
#plt.show()
choice = 1
while(int(choice) == 1):
    momentumtype = raw_input("Enter the type of momentum:\n1. No momentum\n2. Classical Momentum (Polyak's Method)\n3. Nesterov's Accelerated Gradient")
    
    regularization = raw_input("Do you want regularization(Yes - 1, No - 2)")
    
    #Create and object of the Logistic Regression class and initialize with (dataset,labels,tolerance)
    logistic = LR(X,y,1e-7)											
    
    #Call the stochastic gradient descent method with the parameters(Learning_rate,Max iterations,type_of_momentum,regularization_needed_or_not)									
    logistic.calculateStochasticGradientDescent(1e-2,1e4,int(momentumtype),int(regularization))
    
    print "-----For Hinge Loss-----"
    
    #Create and object of the Hinge Loss class and initialize with (dataset,labels,tolerance)
    hingeloss = HL(X,y_hl,1e-7)
    
    #Call the stochastic gradient descent method with the parameters(Learning_rate,Max iterations,type_of_momentum,regularization_needed_or_not)
    hingeloss.calculateStochasticGradientDescent(1e-2,1e4,int(momentumtype),int(regularization))

    choice = raw_input("Do you want to run again(Yes - 1, No - 2)")
