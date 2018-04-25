from numpy import zeros, array, pi, sin, cos, amax, sqrt, dot, append
from numpy.random import rand, choice, randn
from numpy.linalg import norm
def rotation_matrix(axisArr, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axisArr = axisArr / sqrt(dot(axisArr, axisArr))
    a = cos(theta / 2.0)
    b, c, d = -axisArr * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

class network3d(object):
    def __init__(n, profilinActinConc = 5.0, arp23Conc = 50e-3, kBranch = 1.0, cpConc = 50e-3, totalTime = 20.0):
        # Define constants and copy argument values.
        n.kPol = 4.4 * profilinActinConc
        n.kArp23Load = 160.0 * arp23Conc
        n.kBranch = kBranch
        n.kCap = 3.5 * cpConc
        n.totalTime = totalTime

        # Initialize variables.
        n.t = 0.0
        n.dt = 1e-3
        n.N = 200
        n.d = 2.7
        n.w = 10 * n.d
        n.L = 1000.0
        n.zEdge = 0.0
        n.xyzBarbArr = zeros((n.N, 3))
        n.xyzBarbArr[:, 0] = n.L * rand(n.N)
        n.xyzBarbArr[:, 1] = n.L * rand(n.N)
        n.thetaPhiArr = zeros((n.N, 2))
        n.thetaPhiArr[:, 0] = 0.5 * pi * rand(n.N)
        n.thetaPhiArr[:, 1] = 2 * pi * rand(n.N)
        n.dxyzArr = zeros((n.N, 3))
        n.dxyzArr[:, 0] = n.d * sin(n.thetaPhiArr[:, 0]) * cos(n.thetaPhiArr[:, 1])
        n.dxyzArr[:, 1] = n.d * sin(n.thetaPhiArr[:, 0]) * sin(n.thetaPhiArr[:, 1])
        n.dxyzArr[:, 2] = n.d * cos(n.thetaPhiArr[:, 0])
        n.isCappedArr = zeros(n.N, dtype=bool)
        
    def elongate(n, idx):
        n.xyzBarbArr[idx, :] += n.dxyzArr[idx, :]
        
    def branch(n, idx):
        # Generate rotation angles.
        alpha = choice([-1, 1]) * (5.0 / 180 * pi * randn() + 70.0 / 180 * pi)
        beta = 2 * pi * randn()
        # Find a unit vector perpendicular to the direction of the barbed end.
        perpBranchRow = array([1.0, 1.0, (-n.dxyzArr[idx, 0] - n.dxyzArr[idx, 1]) / n.dxyzArr[idx, 2]])
        perpBranchRow = perpBranchRow / norm(perpBranchRow)
        # Perform rotations.
        dxyzRot1Arr = dot(rotation_matrix(perpBranchRow, alpha), n.dxyzArr[idx, :])
        dxyzRot2Arr = dot(rotation_matrix(n.dxyzArr[idx, :], beta), dxyzRot1Arr)
        # Append new barbed ends to array.
        n.xyzBarbArr = append(n.xyzBarbArr, array([n.xyzBarbArr[idx, :]]), axis = 0)
        n.dxyzArr = append(n.dxyzArr, array([dxyzRot2Arr]), axis = 0)
        n.N += 1
        n.isCappedArr = append(n.isCappedArr, False)
        
    def update(n):
        for i in range(n.N):
            if n.isCappedArr[i] == False:
                if rand() < (n.kCap * n.dt):
                    n.isCappedArr[i] = True
                    continue
                elif rand() < (n.kPol * n.dt):
                    n.elongate(i)
                elif (n.zEdge - n.xyzBarbArr[i, 2]) < n.w:
                    if rand() < (n.kBranch * n.dt):
                        n.branch(i)
            elif n.isCappedArr[i] == True:
                if (n.zEdge - n.xyzBarbArr[i, 2]) < n.w:
                    if rand() < (n.kBranch * n.dt):
                        n.branch(i)
        n.zEdge = amax(n.xyzBarbArr[:, 2])
        n.t += n.dt
        
    def simulate(n):
        while n.t <= n.totalTime:
            n.update()