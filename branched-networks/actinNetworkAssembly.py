# Import libraries
from numpy import array, zeros, pi, cos, sin, mod, int, argmin, append, amax, exp, arctan, abs, logical_and
from numpy.random import rand, poisson, randn, choice

class network(object):
    def __init__(n, kPol = 100.0, kBr = 1.0, kCap = 1.0, extForce = 1e2, totalTime = 10.0):
        # Define constants.
        n.L = 1000.0 # Length of leading edge in nanometers
        n.kPol = kPol # Polymerization rate in subunits per second
        n.kBr = kBr # branch rate in branches per second
        n.kCap = kCap # cap rate in branches per second
        n.d = 2.7 # Width of subunit in nanometers
        n.w = 10 * n.d # Width of branching region.
        n.muTheta = 70.0 / 180 * pi
        n.muSigma = 5.0 / 180 * pi
        n.extForce = extForce # external force in pN.
        n.totalTime = totalTime

        # Initialize.
        n.N = 200 # Initial number of filaments
        n.t = 0.0 # Time in seconds.
        n.dt = 1e-3 # Simulation time interval in seconds
        n.tTotal = 10.0 # Total simulation time in seconds
        n.xPointArr = zeros(n.N) # x-coordinate of barbed end position
        n.yPointArr = n.L * rand(n.N) # y-coordinate of barbed end position
        n.thetaArr = pi * rand(n.N) - pi / 2 # angle of barbed end with respect to x-axis
        n.uArr = n.d * cos(n.thetaArr) # x-coordinate of theta
        n.vArr = n.d * sin(n.thetaArr) # y-coordinate of theta
        n.xBarbArr = n.xPointArr
        n.yBarbArr = n.yPointArr
        n.xLead = amax(n.xBarbArr)
        n.isCappedArr = zeros(n.N, dtype = bool)
        n.isActiveArr = n.xLead - n.xBarbArr <= n.w
        
    def cap(n, index):
        n.isCappedArr[index] = True
    
    def branch(n, index):
        theta = choice([-1, 1]) * n.muTheta + n.muSigma * randn()
        u = n.uArr[index]
        v = n.vArr[index]
        uNew = u * cos(theta) - v * sin(theta)
        # Make sure branch points towards the leading edge.
        if uNew > 0:
            vNew = u * sin(theta) + v * cos(theta)
        else:
            uNew = u * cos(theta) + v * sin(theta)
            vNew = -u * sin(theta) + v * cos(theta)
        # Add pointed end to arrays.
        n.xPointArr = append(n.xPointArr, n.xBarbArr[index])
        n.yPointArr = append(n.yPointArr, n.yBarbArr[index])
        n.xBarbArr = append(n.xBarbArr, n.xBarbArr[index])
        n.yBarbArr = append(n.yBarbArr, n.yBarbArr[index])
        n.uArr = append(n.uArr, uNew)
        n.vArr = append(n.vArr, vNew)
        n.isCappedArr = append(n.isCappedArr, False)
        
    def elongate(n, index):
        n.xBarbArr[index] = n.xBarbArr[index] + n.uArr[index]
        n.yBarbArr[index] = n.yBarbArr[index] + n.vArr[index]
        # Enforce periodic boundary conditions in the y-direction.
        if n.yBarbArr[index] > n.L:
            n.yBarbArr[index] = mod(n.yBarbArr[index], n.L)
            n.yPointArr[index] = n.yBarbArr[index]
        if n.yBarbArr[index] < 0:
            n.yBarbArr[index] = mod(n.yBarbArr[index], n.L)
            n.yPointArr[index] = n.yBarbArr[index]
            
    def orderparameter(n):
        thetaArr = arctan(n.vArr / n.uArr) / pi * 180
        n10 = sum(abs(thetaArr) <= 10.0)
        n2040 = sum(logical_and(thetaArr >= -40.0, thetaArr <= -20.0)) + sum(logical_and(thetaArr >= 20.0, thetaArr <= 40.0))
        phi = (n10 - 0.5 * n2040) / (n10 + 0.5 * n2040)
        return phi
                
    def update(n):
        n.N = len(n.xBarbArr)
        n.isActiveArr = n.xLead - n.xBarbArr <= n.w
        n.isTouchingArr = n.xLead - n.xBarbArr < n.d
        n.nTouching = sum(n.isTouchingArr)
        n.forceWeight = exp(-n.extForce * n.d / 4.114 / n.nTouching)
        for i in range(n.N):
            if n.isCappedArr[i] == 0:
                # Cap.
                if n.isTouchingArr[i] == False:
                    capProb = poisson(n.kCap * n.dt)
                else:
                    capProb = poisson(n.kCap * n.forceWeight * n.dt)
                if bool(capProb) == True:
                    n.cap(i)
                    continue
                # Branch.
                if bool(poisson(n.kBr * n.isActiveArr[i] * n.dt)) == True:
                    n.branch(i)
                # Elongate.
                if n.isTouchingArr[i] == False:
                    polProb = poisson(n.kPol * n.isActiveArr[i] * n.dt)
                else:
                    polProb = poisson(n.kPol * n.forceWeight * n.dt)
                if bool(polProb) == True:
                    n.elongate(i)
        n.xLead = amax(n.xBarbArr)
        n.t = n.t + n.dt
        
    def simulate(n):
        while n.t <= n.totalTime and sum(n.isActiveArr) > 0:
            n.update()