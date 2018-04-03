# Import libraries
from numpy import array, zeros, nan, isnan, pi, cos, sin, mod, int, argmin, append, amax, exp, arctan, abs, logical_and, linspace, sqrt
from numpy.random import rand, poisson, randn, choice

class network(object):
    def __init__(n, kPol = 50.0, kBr = 0.5, kCap = 0.5, kAct = 100.0, extForce = 1e2, totalTime = 10.0):
        # Define constants.
        n.L = 1000.0 # Length of leading edge in nanometers
        n.kPol = kPol # Polymerization rate in subunits per second
        n.kBr = kBr # branch rate in branches per second
        n.kCap = kCap # cap rate in branches per second
        n.kAct = kAct # actin loading rate in subunits per second
        n.kTrans = n.kAct / 10.0 # Transfer rate from polyproline to WH2 domain in subunits per second. 
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
        n.xLead = amax(n.xBarbArr[logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)])
        n.isCappedArr = zeros(n.N, dtype = bool)
        n.isTouchingArr = n.xLead - n.xBarbArr < n.d
        n.nTouching = sum(n.isTouchingArr)
        n.forceWeight = exp(-n.extForce * n.d / 4.114 / n.nTouching)
        
        # NPFs.
        n.nNpfs = 50 # Assuming 19x8-um footprint for WAVE complexes.
        n.xNpfArr = zeros(n.nNpfs)
        n.yNpfArr = linspace(0.0, n.L, n.nNpfs)
        n.isWH2LoadedArr = zeros(n.nNpfs, dtype = bool)
        n.isPolProLoadedArr = zeros(n.nNpfs, dtype = bool)
        
    def findbarb(n):
        # Index active barbed ends.
        idxNearBarbArr = zeros(n.nNpfs)
        for i in range(n.nNpfs):
            iDistanceArr = sqrt((n.xBarbArr - n.xNpfArr[i])**2 + (n.yBarbArr - n.yNpfArr[i])**2)
            if any(iDistanceArr <= n.w):
                idxNearBarbArr[i] = argmin(iDistanceArr)
                if n.isCappedArr[int(idxNearBarbArr[i])] == True:
                    idxNearBarbArr[i] = nan
            else:
                idxNearBarbArr[i] = nan
        return idxNearBarbArr
                    
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
            
    def orderparameter(n):
        thetaArr = arctan(n.vArr / n.uArr) / pi * 180
        n10 = sum(abs(thetaArr) <= 10.0)
        n2040 = sum(logical_and(thetaArr >= -40.0, thetaArr <= -20.0)) + sum(logical_and(thetaArr >= 20.0, thetaArr <= 40.0))
        phi = (n10 - 0.5 * n2040) / (n10 + 0.5 * n2040)
        return phi
                
    def update(n):
        # Reactions from solution.
        # Elongate.
        for i in range(n.N):
            if n.isCappedArr[i] == False:
                if n.isTouchingArr[i] == True:
                    polProb = poisson(n.kPol * n.forceWeight * n.dt)
                else:
                    polProb = poisson(n.kPol * n.dt)
                if bool(polProb) == True:
                    n.elongate(i)
                    
        n.xLead = amax(n.xBarbArr[logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)])
        n.xNpfArr[:] = n.xLead
        n.isTouchingArr = n.xLead - n.xBarbArr < n.d
        n.nTouching = sum(n.isTouchingArr)
        n.forceWeight = exp(-n.extForce * n.d / 4.114 / n.nTouching)
        
        # Load NPFs.
        for i in range(n.nNpfs):
            if n.isPolProLoadedArr[i] == False:
                n.isPolProLoadedArr[i] = bool(poisson(n.kAct * n.dt))
        
        # Cap.
        for i in range(n.N):
            if n.isCappedArr[i] == 0:
                if n.isTouchingArr[i] == False:
                    capProb = poisson(n.kCap * n.dt)
                else:
                    capProb = poisson(n.kCap * n.forceWeight * n.dt)
                if bool(capProb) == True:
                    n.cap(i)    
        
        # Polyproline-dependent processes
        n.idxNearBarbArr = n.findbarb()
        for i in range(n.nNpfs):
            if n.isPolProLoadedArr[i] == True:
                # Transfer.
                if n.isWH2LoadedArr[i] == False:
                    if bool(poisson(n.kTrans * n.dt)) == True:
                        n.isPolProLoadedArr[i] == False
                        n.isWH2LoadedArr[i] == True
                # Elongate.
                idxBarb = n.idxNearBarbArr[i]
                if isnan(idxBarb) == False:
                    idxBarb = int(idxBarb)
                    if n.isTouchingArr[idxBarb] == False:
                        polProb = poisson(3 * n.kPol * n.dt)
                    else:
                        polProb = poisson(3 * n.kPol * n.forceWeight * n.dt)
                    if bool(polProb) == True:
                        n.elongate(idxBarb)
                        n.isPolProLoadedArr[i] = False
        
        n.xLead = amax(n.xBarbArr[logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)])
        n.xNpfArr[:] = n.xLead
        n.isTouchingArr = n.xLead - n.xBarbArr < n.d
        n.nTouching = sum(n.isTouchingArr)
        n.forceWeight = exp(-n.extForce * n.d / 4.114 / n.nTouching)
                          
        # WH2-dependent processes.
        n.idxNearBarbArr = n.findbarb()
        for i in range(n.nNpfs):
            idxBarb = n.idxNearBarbArr[i]
            if isnan(idxBarb) == False:
                idxBarb = int(idxBarb)
                # Branch
                if bool(poisson(n.kBr * n.dt)) == True:
                    n.branch(idxBarb)
                    n.isWH2LoadedArr[i] = False
                    continue
                # Elongate
                if n.isTouchingArr[idxBarb] == False:
                    polProb = poisson(3 * n.kPol * n.dt)
                else:
                    polProb = poisson(3 * n.kPol * n.forceWeight * n.dt)
                if bool(polProb) == True:
                    n.elongate(idxBarb)
                    n.isWH2LoadedArr[i] = False
                                
        n.t = n.t + n.dt
        n.N = len(n.xBarbArr)
        n.xLead = amax(n.xBarbArr[logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)])
        n.xNpfArr[:] = n.xLead
        n.isTouchingArr = n.xLead - n.xBarbArr < n.d
        n.nTouching = sum(n.isTouchingArr)
        n.forceWeight = exp(-n.extForce * n.d / 4.114 / n.nTouching)
        
    def simulate(n):
        while n.t <= n.totalTime and sum(~n.isCappedArr) > 0:
            n.update()