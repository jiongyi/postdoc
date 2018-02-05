# Import libraries
from numpy.random import randint, normal, uniform, poisson, choice, exponential
from numpy import array, pi, linspace, argmax, sin, cos, amin, amax, delete, empty, concatenate, sort, logical_and, flatnonzero, arctan2, append, nan, arange, ones, zeros, sum, exp, invert
from scipy import stats

# Function definitions
def bimodal(dx, sigma, nSize = 1):
    """Returns a 1 x nSize array with random numbers sampled from a superposition of two gaussians of standard deviation sigma centered at -dx, +dx."""
    choices = randint(2, size = nSize)
    xDist = array([normal(loc = -dx, scale = sigma) if i == 0 else normal(loc = dx, scale = sigma) for i in choices])
    if nSize > 1:
        return xDist
    else:
        return xDist[0]
    
def mode(xArray, binRes = 100):
    kde = stats.gaussian_kde(xArray)
    bins = linspace(-pi, pi, binRes)
    pdfx = kde(bins)
    return bins[argmax(pdfx)]

class network(object):
    def __init__(n, polRate, branchRate, capRate, loadRate):
        # Copy argument values.
        n.polRate = polRate
        n.branchRate = branchRate
        n.capRate = capRate
        n.loadRate = loadRate
        
        # Define constants.
        n.lamWidth = 1000.0
        n.monomerWidth = 2.7
        n.noBarbed = 200
        n.noNPFs = 30
        n.rxnWidth = 2 * n.monomerWidth
        
        # Initialize variables.
        n.noBranches = 0
        n.noCaps = 0
        n.isCappedArr = zeros(n.noBarbed)
        n.noFilled = 0
        n.noFilledArr = empty([])
        n.tElapsed = 0.0
        n.xEdge = array([0.0])
        n.noFilaments = [[n.noBarbed, n.noBranches, n.noCaps]]
        n.filamentLengthArr = n.monomerWidth * ones(n.noBarbed)
        
        # Initialize.
        # Sample from uniform distribution given N filaments.
        n.thetaArr = uniform(-pi / 2.0, pi / 2.0, size = n.noBarbed)
        n.xArr = uniform(low = 0.0, high = n.lamWidth, size = n.noBarbed)
        n.yArr = uniform(low = -n.rxnWidth, high = 0.0, size = n.noBarbed)
        n.uArr = n.monomerWidth * sin(n.thetaArr)
        n.vArr = n.monomerWidth * cos(n.thetaArr)
        
        n.xArr -= amin(n.xArr)
        n.xBoundary = amax(n.xArr)
        n.noActiveArr = array([n.noBarbed])
        
    def xPeriodic(n, xArr):
        """Enforces periodic boundary conditions in the x-direction"""
        xArr %= n.xBoundary
        return xArr
            
    def elongate(n, idx):
        """Moves barbed ends by one monomer per time step"""
        n.xArr[idx] = n.xPeriodic(n.xArr[idx] + n.uArr[idx])
        n.yArr[idx] += n.vArr[idx]
        n.noFilled -= 1
        n.filamentLengthArr[idx] += n.monomerWidth
        
    def cap(n, idxArr):
        """Saves capping locations"""
        n.isCappedArr[idxArr] = True
        n.noCaps += 1
        n.noBarbed -= 1
        
    def branch(n, idx):
        theta = bimodal(70 / 180 * pi, 5 / 180 * pi)
        u = n.uArr[idx]
        v = n.vArr[idx]
        vPrime = u * sin(theta) + v * cos(theta)
        if vPrime > 0:
            uPrime = u * cos(theta) - v * sin(theta)
        else:
            vPrime = -u * sin(theta) + v * cos(theta)
            uPrime = u * cos(theta) + v * sin(theta)
        
        xPrime = n.xPeriodic(n.xArr[idx] + uPrime)
        yPrime = n.yArr[idx] + vPrime
        n.xArr = append(n.xArr, xPrime)
        n.yArr = append(n.yArr, yPrime)
        n.uArr = append(n.uArr, uPrime)
        n.vArr = append(n.vArr, vPrime)
        n.isCappedArr = append(n.isCappedArr, False)
        n.filamentLengthArr = append(n.filamentLengthArr, n.monomerWidth)
        n.noFilled -= 1
        n.noBarbed += 1
        n.noBranches += 1
        
    def computeweight(n):
        distanceToEdgeArr = amax(n.yArr) - n.yArr
        isAtEdgeArr = distanceToEdgeArr <= n.monomerWidth
        forceArr = 25.0 / sum(isAtEdgeArr) * n.vArr
        forceArr[invert(isAtEdgeArr)] = 0.0
        weightArr = exp(-forceArr * n.monomerWidth / 4.114)
        return weightArr
        
    def indexactive(n):
        isAheadArr = n.yArr >= (n.xEdge[-1] - n.rxnWidth)
        idxActiveArr = flatnonzero(isAheadArr)
        return idxActiveArr
    
    def timeStep(n, dt):
        # Index active filaments.
        idxActiveArr = n.indexactive()
        n.noActive = len(idxActiveArr)
        n.noActiveArr = append(n.noActiveArr, n.noActive)
        
        # Compute force-dependent weights.
        weightArr = n.computeweight()
        
        # Cap.
        """
        noCaps = poisson(n.capRate * n.noActive * dt)
        if noCaps > 0:
            if noCaps >= n.noActive:
                idxCapArr = arange(n.noActive)
            else:
                idxCapArr = choice(idxActiveArr, size = noCaps, replace = False)
            n.cap(idxCapArr)
            n.noCaps += noCaps
            n.noBarbed -= noCaps
        """
        
        # Update NPF occupancy.
        n.noFilled += poisson(n.loadRate * (n.noNPFs - n.noFilled) * dt)
        n.noFilled = min(n.noNPFs, n.noFilled)
        n.noFilledArr = append(n.noFilledArr, n.noFilled)
        
        # Update occupancy array.
        if n.noFilled >= n.noActive:
            isLoadedArr = ones(n.noActive)
        else:
            isLoadedArr = zeros(n.noActive)
            idxLoadArr = choice(arange(n.noActive), size = n.noFilled, replace = False)
            isLoadedArr[idxLoadArr] = True
            
        # Iterate over active barbed ends.
        for i in idxActiveArr:
            capTime = exponential(1 / n.capRate / weightArr[i])
            polTime = exponential(1 / n.polRate / weightArr[i])
            branchTime = exponential(1 / n.branchRate)
            dtTime = exponential(dt)
            if capTime <= dtTime:
                n.cap(i)
                if isLoadedArr[idxActiveArr == i]:
                    if branchTime <= dtTime:
                        n.branch(i)
            else:
                if isLoadedArr[idxActiveArr == i]:
                    if (branchTime < polTime) and (branchTime <= dtTime):
                        n.branch(i)
                    else:
                        if (polTime < branchTime) and (polTime <= dtTime):
                            n.elongate(i)
        
        
        """
        for i in idxActiveArr:
            if isLoadedArr[idxActiveArr == i]:
                branchTime = exponential(1 / n.branchRate)
                polTime = exponential(1 / n.polRate / weightArr[i])
                dtTime = exponential(dt)
                if n.isCappedArr[i] == True:
                    if branchTime <= dtTime:
                        n.branch(i)
                else:
                    if (branchTime < polTime) and (branchTime <= dtTime):
                        n.branch(i)
                    else:
                        if (polTime < branchTime) and (polTime <= dtTime):
                            n.elongate(i)
        """
                               
                            
        # Update.
        n.xEdge = append(n.xEdge, amax(n.yArr))
        n.tElapsed += dt
        
    def evolve(n, dt, tFinal):
        while n.tElapsed < tFinal:
            n.timeStep(dt)
            n.noFilaments += [[n.noBarbed, n.noBranches, n.noCaps]]
            
    def getAngles(n):
        return arctan2(n.uArr, n.vArr)
            
        
        
    
    