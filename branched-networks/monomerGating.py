# Import libraries
from numpy.random import randint, normal, uniform, poisson, choice, exponential
from numpy import array, pi, linspace, argmax, sin, cos, amin, amax, delete, empty, concatenate, sort, logical_and, flatnonzero, arctan2, append, nan, arange, ones, zeros, sum, exp
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
    def __init__(n, polRate, branchRate, capRate, loadRate, noFilaments, noNPFs, rxnWidth, recordHistory = False):
        n.polRate = polRate
        n.noBarbed = noFilaments
        n.branchRate = branchRate
        n.noBranches = 0
        n.capRate = capRate
        n.noCaps = 0
        n.isCappedArr = zeros(n.noBarbed)
        n.loadRate = loadRate
        n.noNPFs = noNPFs
        n.rxnWidth = rxnWidth
        n.noFilled = 0
        n.noFilledArr = empty([])
        n.tElapsed = 0.0
        n.xEdge = array([0.0])
        n.noFilaments = [[n.noBarbed, n.noBranches, n.noCaps]]
        n.Fext = 0.0
        
        # Initialize.
        filRange = 1000.0
        n.monoWidth = 2.7
        # Sample from uniform distribution given N filaments.
        n.thetaArr = uniform(-pi / 2.0, pi / 2.0, size = noFilaments)
        n.xArr = uniform(low = 0.0, high = filRange, size = noFilaments)
        n.yArr = uniform(low = -rxnWidth, high = 0.0, size = noFilaments)
        n.uArr = n.monoWidth * sin(n.thetaArr)
        n.vArr = n.monoWidth * cos(n.thetaArr)
        
        n.xArr -= amin(n.xArr)
        n.xBoundary = amax(n.xArr)
        n.noActive = noFilaments
    def xPeriodic(n, xArr):
        """Enforces periodic boundary conditions in the x-direction"""
        xArr %= n.xBoundary
        return xArr
    
    def computeforce(n, idx):
        """Finds out force on filament"""
        theta = arctan2(n.yArr[idx] / n.xArr[idx])
        cosTheta = cos(theta)
        
    def elongate(n, idx):
        """Moves barbed ends by one monomer per time step"""
        n.xArr[idx] = n.xPeriodic(n.xArr[idx] + n.uArr[idx])
        n.yArr[idx] += n.vArr[idx]
        n.noFilled -= 1
    def cap(n, idxArr):
        """Saves capping locations and removes barbed ends from position and orientation arrays
        n.xArr = delete(n.xArr, idxArr)
        n.yArr = delete(n.yArr, idxArr)
        n.uArr = delete(n.uArr, idxArr)
        n.vArr = delete(n.vArr, idxArr)"""
        n.isCappedArr[idxArr] = True
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
        n.noFilled -= 1
        n.noBarbed += 1
        n.noBranches += 1
        
    def indexactive(n):
        isBehindArr = n.xEdge[-1] > n.yArr
        isAheadArr = n.yArr >= (n.xEdge[-1] - n.rxnWidth)
        isActiveArr = logical_and(isBehindArr, isAheadArr)
        idxActiveArr = flatnonzero(isActiveArr)
        return idxActiveArr
    
    def timeStep(n, dt):
        # Index active filaments.
        idxActiveArr = n.indexactive()
        n.noActive = len(idxActiveArr)
        
        # Calculate force.
        cosThetaArr = cos(arctan2(n.yArr[idxActiveArr], n.xArr[idxActiveArr]))
        sumCosThetaArr = sum(cosThetaArr)
        forceActiveArr = n.Fext * cosThetaArr / sumCosThetaArr
        expFactorArr = exp(-forceActiveArr * 2.7 / 4.114)
        
        # Cap.
        noCaps = poisson(n.capRate * n.noActive * dt)
        if noCaps > 0:
            if noCaps >= n.noActive:
                idxCapArr = arange(n.noActive)
            else:
                idxCapArr = choice(idxActiveArr, size = noCaps, replace = False)
            n.cap(idxCapArr)
            n.noCaps += noCaps
            n.noBarbed -= noCaps
        
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
            if isLoadedArr[idxActiveArr == i]:
                branchTime = exponential(1 / n.branchRate / expFactorArr[idxActiveArr == i])
                polTime = exponential(1 / n.polRate / expFactorArr[idxActiveArr == i])
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
                               
                            
        # Update.
        n.xEdge = append(n.xEdge, amax(n.yArr))
        n.tElapsed += dt
        
    def evolve(n, dt, tFinal):
        while n.tElapsed < tFinal:
            n.timeStep(dt)
            n.noFilaments += [[n.noBarbed, n.noBranches, n.noCaps]]
            
    def getAngles(n):
        return arctan2(n.uArr, n.vArr)
            
        
        
    
    