# Import libraries
#------------------------------------------------------------------------------------------------------------------------------
from numpy import array,matrix,arange,cumsum,linspace,append,gradient,delete,all,in1d,histogram,around,amax,amin,mean,cos,sin,cosh,arctan,arctan2,exp,log,pi,meshgrid,median,argmax,NaN
from scipy.stats import kurtosis
from numpy.random import poisson,normal,randint,uniform,binomial
from numpy.linalg import norm
from scipy.signal import savgol_filter
from matplotlib.pyplot import figure,subplots,savefig,close,plot,subplot,axes,axis,xlim,xlabel,ylim,ylabel,legend,contourf
from os import system,getcwd

# Function definitions
#------------------------------------------------------------------------------------------------------------------------------
def heavisidePi(x, dx):
    if abs(x) <= dx / 2.0:
        return 1.0
    else:
        return 0.0

def heavisideTheta(x):
    if 0 <= x:
        return 1.0
    else:
        return 0.0
    
def bimodal(dx, sigma, nSize = 1):
    """Returns a 1 x nSize array with random numbers sampled from a superposition of two gaussians of standard deviation sigma centered at -dx, +dx."""
    choices = randint(2, size = nSize)
    xDist = array([normal(loc = -dx, scale = sigma) if i == 0 else normal(loc = dx, scale = sigma) for i in choices])
    if nSize > 1:
        return xDist
    else:
        return xDist[0]

def mode(xArray, binRes = 100):
    hist, bins = histogram(xArray, bins = binRes)
    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers[argmax(hist)]

# Main class object.
#------------------------------------------------------------------------------------------------------------------------------
class network(object):
    # Initialization
    def __init__(n, rLamba, rBeta, rKappa, xSeed, dxSeed, branchTheta = 1.3003, branchSigma = 0.1354, forceDirection = True,                              recordHistory = False):
        n.rLambda = rLamba
        n.rBeta = rBeta
        n.rKappa = rKappa
        n.branchTheta = branchTheta
        n.branchSigma = branchSigma
        # Move network so the offset in the x direction is zero.
        xSeed.T[0] -= amin(xSeed.T[0])
        n.xBoundary = amax(xSeed.T[0])
        # Set network frontier as the list of barbed ends.
        n.Frontier = list(zip(xSeed, dxSeed))
        n.monomerSize = norm(dxSeed[0])
        # Initialize number of events and angle mode
        n.nCapped = 0
        n.nBranched = 0
        n.nBarbed = len(xSeed)
        n.phiMax = abs(mode(n.getAngles(n.Frontier)))
        n.nFilaments = [[n.nBarbed, n.nBranched, n.nCapped, n.phiMax]]
        # Position of leading edge and time elapsed
        n.tElapsed = 0.0
        n.xEdge = 0.0
        # Save force direction and history record settings
        n.forceDirection = forceDirection
        n.recordHistory = recordHistory
        # If history is recorded, record all (x, dx)
        n.Monomers = list(zip(xSeed, dxSeed))
        n.Branches = []
        n.Caps = []

    def xPeriodic(n, rArray):
        """Enforces periodic boundary conditions in the x direction"""
        x = rArray[0]
        y = rArray[1]
        x %= n.xBoundary
        return array([x, y])
    
    def getPositions(n, sites):
        return array([x for x, dx, in sites])
    
    def getAngles(n, sites):
        return array([arctan2(dx[0], dx[1]) for x, dx in sites])
    
    def elongate(n, iIndex):
        """Adds a monomer of width dr to the iIndex-th seed"""
        r, dr = n.Frontier[iIndex]
        n.Frontier[iIndex] = (n.xPeriodic(r + dr), dr)
        if n.recordHistory == True:
            n.Monomers += [(n.xPeriodic(r + dr), dr)]
            
    def branch(n, iIndex):
        r, dr = n.Frontier[iIndex]
        x = dr[0]
        y = dr[1]
        if n.forceDirection == True:
            while True:
                theta = bimodal(dx = n.branchTheta, sigma = n.branchSigma)
                dr = array([x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)])
                if dr[1] > 0:
                    break
        else:
            theta = bimodal(dx = n.branchTheta, sigma = n.branchSigma)
            dr = array([x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)])
        
        n.Frontier += [(r, dr)]
        n.nBranched += 1
        n.nBarbed += 1
        if n.recordHistory == True:
            n.Branches += [(r, dr)]
            
    def cap(n, iIndex):
        r, dr = n.Frontier[iIndex]
        del n.Frontier[iIndex]
        n.nCapped += 1
        n.nBarbed -= 1
        if n.recordHistory == True:
            n.Caps += [(r, dr)]
            
    def timeStep(n, dt, Fext = 0.0):
        lFrontier = n.Frontier
        i = 0
        n.D = n.nBarbed / n.xBoundary
        for x, dx in lFrontier:
            capping = poisson(n.rKappa(x[0], x[1] - n.xEdge, n.tElapsed) * dt)
            if bool(capping) == True:
                n.cap(i)
                continue
            elongating = poisson(n.rLambda(x[0], x[1] - n.xEdge, n.tElapsed) * dt)
            if bool(elongating) == True:
                n.elongate(i)
            branching = poisson(n.rBeta(x[0], x[1] - n.xEdge, n.tElapsed) * dt / n.D)
            if bool(branching) == True:
                n.branch(i)
            i += 1
        n.tElapsed += dt
        v0 = n.rLambda(0.0, 0.0, 0.0) * n.monomerSize
        kT = 4.1
        n.v = v0 * exp(-Fext * n.monomerSize / (kT * n.D))
        n.xEdge += n.v * dt
    
    def evolve(n, dt, tFinal, Fext = 0.0):
        while n.nBarbed != 0 and n.tElapsed <= tFinal:
            n.timeStep(dt, Fext)
            n.phiMax = abs(mode(n.getAngles(n.Frontier)))
            n.nFilaments += [[n.nBarbed, n.nBranched, n.nCapped, n.phiMax]]
    
    def exportData(n, dt, ds, tFinal, Fext = 0.0):
        j = 1
        while n.nBarbed != 0 and n.tElapsed <= tFinal:
            n.plotAngles().savefig("figures/angles"+str(j).zfill(3)+".png"); close();
            if n.recordHistory == True:
                n.plotData().savefig("figures/network"+str(j).zfill(3)+".png"); close();
            n.evolve(dt, n.tElapsed + ds, Fext)
            j += 1
        n.plotFilaments().savefig("output/filamentStatistics.png"); close();
        system("convert -delay 10 -loop 0 figures/angles*.png output/anglesDistribution.gif"); system("rm -R figures/angles*.png");
        if n.recordHistory == True:
            system("convert -delay 10 -loop 0 figures/network*.png output/networkPlot.gif"); system("rm -R figures/network*.png");
    
    def plotData(n):
        # Get particle positions.
        xFil = n.getPositions(n.Monomers)
        xBranch = n.getPositions(n.Branches)
        xCap = n.getPositions(n.Caps)
        # Create coordinate mesh.
        yRange = arange(0.0, n.xBoundary)
        xRange = arange(n.xEdge + 30.0 - n.xBoundary, n.xEdge + 30.0)
        xGrid, yGrid = meshgrid(xRange, yRange)
        zGrid = array([[n.rLambda(Y, X - n.xEdge, 0.0) for X in xRange] for Y in yRange])
        # Figure plotting actin network.
        Plot = figure(figsize = (16, 16))
        
        # Plot particles.
        plot(xFil.T[1], xFil.T[0], 'g', marker = ".", linewidth = 0, ms = 5.5, alpha = 0.5)
        if len(xBranch) != 0:
            plot(xBranch.T[1], xBranch.T[0], '#2737ff', marker = '.', linewidth = 0, ms = 10)
        if len(xCap) != 0:
            plot(xCap.T[1], xCap.T[0], '#ff0000', marker = '.', linewidth = 0, ms = 10)
        
        # Customize plotting options.
        xlabel(r"Distance, $x$ / nm", fontsize = 28)
        ylabel(r"Distance, $y$ / nm", fontsize = 28)
        xlim(n.xEdge + 30.0 - n.xBoundary, n.xEdge + 30.0)
        ylim(0, n.xBoundary)
        axes().set_aspect('equal', 'box')
        axes().tick_params(labelsize = 16)
        
        # Together with rate envelope.
        contourf(xGrid, yGrid, zGrid, cmap = 'Greens', vmin = 140.0, vmax = 141)
        
        # Return figures
        return Plot

    def plotFilaments(n, smoothingWindow = 11, smoothingOrder = 3):
        # Extract filament numbers.
        barbed = (array(n.nFilaments).T[0]).astype(float)
        branched = gradient(array(n.nFilaments).T[1]).astype(float)
        capped = gradient(array(n.nFilaments).T[2]).astype(float)
        # Extract filament angle mode.
        phiMax = (array(n.nFilaments).T[3]).astype(float)
        
        # Normalize to mean.
        barbed /= mean(barbed)
        branched /= mean(branched)
        capped /= mean(capped)
        
        # Maximum
        yMax = max(amax(barbed), amax(branched), amax(capped))
        
        # Time range.
        t = linspace(0.0, n.tElapsed, num = len(barbed))
        
        # Plot number of filaments
        f, ax1 = subplots()
        
        ax1.plot(t, savgol_filter(barbed, smoothingWindow, smoothingOrder), 'g', label = "Barbed Ends")
        ax1.plot(t, barbed, 'g')
        ax1.plot(t, savgol_filter(branched, smoothingWindow, smoothingOrder), 'b', label = "Branching Rate")
        ax1.plot(t, branched, 'b')
        ax1.plot(t, savgol_filter(capped, smoothingWindow, smoothingOrder), 'r', label = "Capping Rate")
        ax1.plot(t, capped, 'r')
        legend()
        
        ax2 = ax1.twinx()
        ax2.plot(t, savgol_filter(180 * phiMax / pi, smoothingWindow, smoothingOrder), 'k')
        ax2.plot(t, 180 * phiMax / pi, 'k')
        
        # Label plots.
        ax1.set_xlabel(r"Time, $t$ / $\sec$", fontsize = 16)
        ax1.set_ylabel(r"Count Variation, $n(t)$", fontsize = 16)
        ax2.set_ylabel(r"Filament angle mode, $\phi(t)$ / $\deg$", fontsize = 16)
        xlim(0, n.tElapsed)
        
        f.set_figheight(6)
        f.set_figwidth(12)
        
        return f
    
    def plotAngles(n):
        # Generate histogram
        angleHistogram = figure(figsize = (8, 8))
        hist, bins = histogram(n.getAngles(n.Frontier), bins = 40, normed = False)
        centers = (bins[1:] + bins[:-1]) / 2.0
        
        # Plot polar histogram.
        ax = subplot(111, projection = "polar")
        ax.bar(centers, hist, color =  'g', width = 2 * pi / 40, edgecolor = "none", align = "center")
        ax.tick_params(labelsize = 16)
        
        # Labels.
        ax.set_xlabel(r"Elongating filament angles, $\phi$ / $^{o}$", fontsize = 28)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction('clockwise')
        ax.set_ylim(0, 25)
        ax.set_yticks(array([0]))
        ax.set_xticks(array([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, NaN, 180]) / 180 * pi)
        return angleHistogram
        
                           
                                           
            
    
