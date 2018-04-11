# Import libraries
from numpy import array, copy, flatnonzero, zeros, nan, isnan, pi, cos, sin, mod, int, argmin, append, amax, exp, arctan, abs, logical_and, linspace, sqrt
from numpy.random import rand, poisson, randn, choice

class network(object):
    def __init__(n, profActConc = 5.0, arp23Conc = 50e-3, kBr = 1.42, capConc = 50e-3, extForce = 0.0, totalTime = 20.0):
        # Define constants.
        n.L = 1000.0 # Length of leading edge in nanometers
        n.kPol = 4.4 * profActConc # Polymerization rate in subunits per second
        n.kBr = kBr # branch rate in branches per second
        n.kCap = 3.5 * capConc # cap rate in branches per second
        n.kActLoad = 2.2 * profActConc # actin loading rate in subunits per second
        n.kArpLoad = 1.0 * arp23Conc # Arp2/3 complex loading rate to NPFs in subunits per second
        n.kTrans = 3.0 # Transfer rate from polyproline to WH2 domain in subunits per second. Assumed this is koff for profilin-actin
        n.kProPol = 3.0
        n.kWH2Pol = 3.0
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
        n.xPointArr = zeros(n.N) # x-coordinate of barbed end position
        n.yPointArr = n.L * rand(n.N) # y-coordinate of barbed end position
        n.thetaArr = pi * rand(n.N) - pi / 2 # angle of barbed end with respect to x-axis
        n.uArr = n.d * cos(n.thetaArr) # x-coordinate of theta
        n.vArr = n.d * sin(n.thetaArr) # y-coordinate of theta
        n.xBarbArr = copy(n.xPointArr) + n.uArr
        n.yBarbArr = copy(n.yPointArr) + n.vArr
        n.isCappedArr = zeros(n.N, dtype = bool)
        
        n.xLead = amax(n.xBarbArr[logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)])
        n.forceWeightArr = n.monomergap()
        
        # NPFs.
        n.nNpfs = 50 # Assuming 19x8-um footprint for WAVE complexes.
        n.xNpfArr = n.xLead * zeros(n.nNpfs)
        n.yNpfArr = linspace(0.0, n.L, n.nNpfs)
        n.isWH2LoadedArr = zeros(n.nNpfs, dtype = bool)
        n.isPolProLoadedArr = zeros(n.nNpfs, dtype = bool)
        n.hasArp23Arr = zeros(n.nNpfs, dtype = bool)
        
    def findbarb(n):
        # Index active barbed ends.
        idxNearBarbArr = zeros(n.nNpfs)
        for i in range(n.nNpfs):
            iDistanceArr = sqrt((n.xBarbArr - n.xNpfArr[i])**2 + (n.yBarbArr - n.yNpfArr[i])**2)
            if any(iDistanceArr <= n.w):
                idxNearBarbArr[i] = argmin(iDistanceArr)
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
        n.xBarbArr[index] += n.uArr[index]
        n.yBarbArr[index] += n.vArr[index]

    def monomergap(n):
        isBehindLeadArr = logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)
        lagArr = n.xLead - n.xBarbArr
        isTouchingArr = logical_and(lagArr <= n.d, isBehindLeadArr)
        #distanceWeightArr = lagArr / n.d
        #distanceWeightArr[~isTouchingArr] = 1.0
        forceWeightArr = exp(-n.extForce * n.d * n.uArr / 4.114 / sum(n.uArr[isTouchingArr]))
        forceWeightArr[~isTouchingArr] = 1.0
        #weightArr = distanceWeightArr * forceWeightArr
        weightArr = forceWeightArr
        return weightArr
        
    def orderparameter(n):
        thetaArr = arctan(n.vArr / n.uArr) / pi * 180
        n10 = sum(abs(thetaArr) <= 10.0)
        n2040 = sum(logical_and(thetaArr >= -40.0, thetaArr <= -20.0)) + sum(logical_and(thetaArr >= 20.0, thetaArr <= 40.0))
        phi = (n10 - 0.5 * n2040) / (n10 + 0.5 * n2040)
        return phi
                
    def update(n):
        n.idxNearBarbArr = n.findbarb()
        # Reactions at barbed ends.
        for i in range(n.N):
            if n.isCappedArr[i] == False:
                if bool(poisson(n.kCap * n.forceWeightArr[i] * n.dt)) == True:
                    n.cap(i)
                    continue
                if bool(poisson(n.kPol * n.forceWeightArr[i] * n.dt)) == True:
                    n.elongate(i)
                        
        # Reactions at NPFs.
        for i in range(n.nNpfs):
            idxBarb = n.idxNearBarbArr[i]
            if isnan(idxBarb) == False:
                idxBarb = int(idxBarb)
            # Polyproline region
            if n.isPolProLoadedArr[i] == False:
                n.isPolProLoadedArr[i] = bool(poisson(n.kActLoad * n.dt)) # Load profilin-actin
            else:
                if n.isWH2LoadedArr[i] == False:
                    if bool(poisson(n.kTrans * n.dt)) == True:
                        n.isPolProLoadedArr[i] = False
                        n.isWH2LoadedArr[i] = True # Transfer actin monomer to WH2 domain.
                else:
                    if isnan(idxBarb) == False:
                        if n.isCappedArr[idxBarb] == False:
                            polProb = poisson(n.kProPol * n.forceWeightArr[idxBarb] * n.dt)
                            if bool(polProb) == True:
                                n.elongate(idxBarb) # Elongate nearest filament.
                                n.isPolProLoadedArr[i] = False
            # WH2 domain
            if n.isWH2LoadedArr[i] == True:
                if isnan(idxBarb) == False:
                    if n.isCappedArr[idxBarb] == False:
                        polProb = poisson(n.kWH2Pol * n.forceWeightArr[idxBarb] * n.dt)
                        if bool(polProb) == True:
                            n.elongate(idxBarb) # Elongate.
                            n.isWH2LoadedArr[i] = False
            # CA domain            
            if n.hasArp23Arr[i] == False:
                n.hasArp23Arr[i] = bool(poisson(n.kArpLoad * n.dt))
            else:
                if isnan(idxBarb) == False:
                    if rand() <= 0.026:
                        if bool(poisson(n.kBr * n.dt)) == True:
                            n.branch(idxBarb) # Branch
                            n.hasArp23Arr[i] = False
                            n.isWH2LoadedArr[i] = False

        # Update network.
        n.t += n.dt
        n.N = len(n.xBarbArr)
        isBehindLeadArr = logical_and(n.yBarbArr <= n.L, n.yBarbArr >= 0)
        n.xLead = max(amax(n.xBarbArr[isBehindLeadArr]), n.xLead)
        n.xNpfArr[:] = n.xLead
        n.forceWeightArr = n.monomergap()
    def simulate(n):
        while n.t <= n.totalTime and sum(~n.isCappedArr) > 0:
            n.update()