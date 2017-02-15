from scipy.integrate import odeint
import numpy as np
import matplotlib.pylab as plt

tauN = 886.7
Q = 1.293 #MeV, mass difference
H1 = 1.13

#def lambdaNP(x):
#    return (255./(tauN*x**5.))*(12. + 6.*x + x**2.)
#    
#def H(x):
#    return 1.13
#    
#def f(Xn, x):
#    y = Xn
#    dXndx = (x*lambdaNP(x)/H(x))*(np.exp(-x) - y*(1 + np.exp(-x)))
#    return dXndx
    
def lambdaNP(x):
    return (255./(tauN*(Q/x)**5.))*(12. + 6.*(Q/x) + (Q/x)**2.)
        
def f(Xn, x):
    y = Xn
    dXndx = ((Q/x)*lambdaNP((Q/x))/H1)*(np.exp(-(Q/x)) - y*(1 + np.exp(-(Q/x))))
    return dXndx
    
Xn0 = [0.9]
T = np.logspace(1,-3, 100)
#T = np.linspace(10, 0.001, 100)
#T = np.linspace(0.001, 10, 100)
#print 1/T
result = odeint(f, Xn0, T)
#print result
#plt.loglog(T, result[:,0])
plt.semilogy(T, result[:,0])
plt.gca().set_xlim([0.02, 1.1])
plt.gca().invert_xaxis()
plt.show()
