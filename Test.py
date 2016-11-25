from LogisticRegression import LogisticRegression as LR
import numpy
import matplotlib.pyplot as plt

x = raw_input("enter x co-ordinate of mean1 ")
y = raw_input("enter y co-ordinate of mean1 ")
mean1 = numpy.array([int(x),int(y)])
x1 = raw_input("enter x1 of the 2-dimensional covariance matrix ")
y1 = raw_input("enter y1 of the 2-dimensional covariance matrix ")
x2 = raw_input("enter x2 of the 2-dimensional covariance matrix ")
y2 = raw_input("enter y2 of the 2-dimensional covariance matrix ")
covariance1 = numpy.array([[int(x1),int(y1)],[int(x2),int(y2)]])
n1 = raw_input("enter the number of samples of first class ")
X1 = numpy.random.multivariate_normal(mean1,covariance1,int(n1))

x = raw_input("enter x co-ordinate of mean2 ")
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

fig = plt.figure()


plt.scatter(X1[:,0],X1[:,1], marker='+')
plt.scatter(X2[:,0],X2[:,1], c= 'green', marker='o')
#plt.show()
choice = 1
while(int(choice) == 1):
    momentumtype = raw_input("Enter the type of momentum:\n1. No momentum\n2. Classical Momentum (Polyak's Method)\n3. Nesterov's Accelerated Gradient")
    regularization = raw_input("Do you want regularization(Yes - 1, No - 2)")
    logistic = LR(X,y,1e-7)
    logistic.calculateStochasticGradientDescent(2e-5,10e6,int(momentumtype),int(regularization))
    choice = raw_input("Do you want to run again(Yes - 1, No - 2)")
plt.show()
