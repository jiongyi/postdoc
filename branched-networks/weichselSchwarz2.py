# Import libraries
from numpy.random import randint, normal, uniform, poisson, choice
from numpy import array, pi, linspace, argmax, sin, cos, amin, amax, delete, empty, concatenate, sort, logical_and, flatnonzero, arctan2, append, nan, arange
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
    """Returns the mode of a distribution"""
    kde = stats.gaussian_kde(xArray)
    bins = linspace(-pi, pi, binRes)
    pdfx = kde(bins)
    return bins[argmax(pdfx)]

class network(object):
    def __init__(n, polRate, branchRate, capRate, noFilFront, rxnWidth):
        # Set constants.
        noFilaments = 200
        filRange = 1000.0
        n.monoWidth = 2.7
        
        # Initialize variables.
        n.polRate = polRate
        n.noBarbed = noFilaments
        n.branchRate = branchRate
        n.noBranches = 0
        n.capRate = capRate
        n.noCaps = 0
        n.noFilFront = noFilFront
        n.tElapsed = 0.0
        n.xEdge = array([0.0])
        n.atanArr = array([])
        n.noFilaments = [[n.noBarbed, n.noBranches, n.noCaps]]
        
        # Set up noFilaments in a box of width rxnWidth and length filRange.
        # Filament orientation angles are sampled from uniform distribution.
        n.thetaArr = uniform(-pi / 2.0, pi / 2.0, size = noFilaments)
        n.xArr = uniform(low = 0.0, high = filRange, size = noFilaments)
        n.yArr = uniform(low = -rxnWidth, high = 0.0, size = noFilaments)
        n.uArr = n.monoWidth * sin(n.thetaArr)
        n.vArr = n.monoWidth * cos(n.thetaArr)
        
        n.xArr -= amin(n.xArr)
        n.xBoundary = amax(n.xArr)
        n.noActive = noFilaments - noFilFront
    
    def xPeriodic(n, xArr):
        """Enforces periodic boundary conditions in the x-direction"""
        xArr %= n.xBoundary
        return xArr
    
    def elongate(n, idxArr):
        """Moves barbed ends by one monomer per time step"""
        n.xArr[idxArr] = n.xPeriodic(n.xArr[idxArr] + n.uArr[idxArr])
        n.yArr[idxArr] += n.vArr[idxArr]
    
    def cap(n, idxArr):
        """Saves capping locations and removes barbed ends from position and orientation arrays"""
        n.xArr = delete(n.xArr, idxArr)
        n.yArr = delete(n.yArr, idxArr)
        n.uArr = delete(n.uArr, idxArr)
        n.vArr = delete(n.vArr, idxArr)
    
    def branch(n, idxArr):
        arrLength = len(idxArr)
        uPrimeArr = empty(arrLength)
        vPrimeArr = empty(arrLength)
        for i in range(0, arrLength):
            theta = bimodal(70 / 180 * pi, 5 / 180 * pi)
            u = n.uArr[idxArr[i]]
            v = n.vArr[idxArr[i]]
            vPrime = u * sin(theta) + v * cos(theta)
            if vPrime > 0:
                uPrime = u * cos(theta) - v * sin(theta)
            else:
                vPrime = -u * sin(theta) + v * cos(theta)
                uPrime = u * cos(theta) + v * sin(theta)
            vPrimeArr[i] = vPrime
            uPrimeArr[i] = uPrime
        
        n.xArr = concatenate((n.xArr, n.xArr[idxArr]))
        n.yArr = concatenate((n.yArr, n.yArr[idxArr]))
        n.uArr = concatenate((n.uArr, uPrimeArr))
        n.vArr = concatenate((n.vArr, vPrimeArr))
        n.atanArr = concatenate((n.atanArr, arctan2(uPrimeArr, vPrimeArr)))
    
    def indexactive(n):
        """Returns index of filaments that are active"""
        if n.noFilFront >= n.noBarbed:
            return array([])
        else:
            ySortedArr = sort(n.yArr)
            yLastFront = ySortedArr[-1 - n.noFilFront]
            isBehindArr = n.yArr <= yLastFront
            isAheadArr = n.yArr >= (yLastFront - 2 * n.monoWidth)
            isActiveArr = logical_and(isBehindArr, isAheadArr)
            idxActiveArr = flatnonzero(isActiveArr)
            return idxActiveArr
    
    def timeStep(n, dt):
        # Index active filaments.
        idxActiveArr = n.indexactive()
        n.noActive = len(idxActiveArr)
        
        if n.noActive > 0:
            # Polymerize
            n.elongate(idxActiveArr)
            n.xEdge = append(n.xEdge, amax(n.yArr))
            
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
            
            # Branch.
            idxActiveArr = n.indexactive()
            n.noActive = len(idxActiveArr)
            if n.noActive > 0:
                noBranches = poisson(n.branchRate * dt)
                if noBranches > 0:
                    if noBranches >= n.noActive:
                        idxBranchArr = arange(n.noActive)
                        n.noBranches += n.noActive
                        n.noBarbed += n.noActive
                    else:
                        idxBranchArr = choice(idxActiveArr, size = noBranches, replace = False)
                        n.noBranches += noBranches
                        n.noBarbed += noBranches
                    n.branch(idxBranchArr)
            
        # Update.
        n.tElapsed += dt
        
    def evolve(n, dt, tFinal):
        while n.noBarbed >= 0 and n.noActive >= 0 and n.tElapsed <= tFinal:
            n.timeStep(dt)
            n.noFilaments += [[n.noBarbed, n.noBranches, n.noCaps]]
            
    def getAngles(n):
        return arctan2(n.uArr, n.vArr)