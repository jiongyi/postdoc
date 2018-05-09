# Import libraries
from numpy import array, copy, sort, flatnonzero, zeros, pi, cos, sin, mod, argmin, append, amax, exp, arctan, logical_and, linspace, sqrt
from numpy.random import rand, randn, choice

class network(object):
    def __init__(n, kBr = 20.0, kCap = 0.05, noFilFront = 2, totalTime = 20.0):
        # Define constants.
        n.L = 1000.0 # Length of leading edge in nanometers
        n.kPol = 900 # Elongation rate in nanometers per second
        n.kBr = kBr # branch rate in branches per second
        n.kCap = kCap # cap rate in branches per second
        n.d = 2.7 # Width of subunit in nanometers
        n.w = 2 * n.d # Width of branching region.
        n.muTheta = 70.0 / 180 * pi
        n.muSigma = 5.0 / 180 * pi
        n.noFilFront = noFilFront # external force in pN.
        n.totalTime = totalTime

        # Initialize.
        n.N = 200 # Initial number of filaments
        n.t = 0.0 # Time in seconds.
        n.dt = 1e-3 # Simulation time interval in seconds
        n.xPointArr = zeros(n.N) # x-coordinate of barbed end position
        n.yPointArr = n.L * rand(n.N) # y-coordinate of barbed end position
        n.thetaArr = pi * rand(n.N) - pi / 2 # angle of barbed end with respect to x-axis
        n.uArr = cos(n.thetaArr) # x-coordinate of theta
        n.vArr = sin(n.thetaArr) # y-coordinate of theta
        n.xBarbArr = copy(n.xPointArr) + n.uArr
        n.yBarbArr = copy(n.yPointArr) + n.vArr
        n.isCappedArr = zeros(n.N, dtype = bool)
        n.xLead = amax(n.xBarbArr)

    def findbarb(n):
        # Index active barbed ends.
        if n.noFilFront >= sum(~n.isCappedArr):
            return array([])
        else:
            xSortedArr = sort(n.xBarbArr)
            xLastFront = xSortedArr[-1 - n.noFilFront]
            isBehindArr = n.xBarbArr < xLastFront
            isAheadArr = n.xBarbArr >= (xLastFront - n.w)
            return flatnonzero(logical_and(isBehindArr, isAheadArr))
                    
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
        n.xBarbArr[index] += (n.uArr[index] * n.kPol * n.dt)
        n.yBarbArr[index] += (n.vArr[index] * n.kPol * n.dt)
        # Enforce periodic boundary conditions in the y direction.
        if n.yBarbArr[index] > n.L:
            n.yBarbArr[index] = n.yBarbArr[index] - n.L
            n.yPointArr[index] = 0.0
            n.xPointArr[index] = n.xBarbArr[index]
        if n.yBarbArr[index] < 0.0:
            n.yBarbArr[index] = n.yBarbArr[index] + n.L
            n.yPointArr[index] = n.L
            n.xPointArr[index] = n.xBarbArr[index]
                
    def update(n):
        # Find active barbed ends.
        n.idxActiveBarbArr = n.findbarb()
        # Normalize branching rate.
        kBrCurrent = n.kBr / len(n.idxActiveBarbArr)
        if len(n.idxActiveBarbArr) > 0:
            # Iterate over active barbed ends.
            for idx in n.idxActiveBarbArr:
                if n.isCappedArr[idx]:
                    continue
                else:
                    # Elongate deterministically.
                    n.elongate(idx)
                    # Check branching.
                    if rand() <= (kBrCurrent * n.dt):
                        n.branch(idx)
                    elif rand() <= (n.kCap * n.dt):
                        n.cap(idx)
                   
        # Update network.
        n.t += n.dt
        n.N = len(n.xBarbArr)
        n.xLead = amax(n.xBarbArr)
        
    def simulate(n):
        while n.t <= n.totalTime:
            n.update()