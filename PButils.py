# a collection of support utilities
# to aid analysis of Poisson-Boltzmann systems
# e.g. calculating Debye length, surface charge etc

import scipy.constants as sc
import numpy
from enum import Enum

water_EPS_0_25dC = 78.36 #Water at 25dC
EPS_0 = 66.14 # Propylene Carbonate

pKwaterDiss = 14

# buffer and other solutions adjust counterions depending on deviation
# from neutral pH.  Pure neutral pH is 7, but that's without dissolved CO2
# 2004 CO2 levels around 375-377 ppm, CRC 2010, leads to naturalpH = 5.6
neutralpH=7.0     # "natural pH" without dissolved CO2


# enumerate types of boundary condition
class BoundaryConditionType(Enum):
    CONSTANT_POTENTIAL = 1
    CONSTANT_CHARGE = 2
    CHARGE_REGULATED = 3
    CONSTANT_POTENTIAL_DIFFERENCE = 4


# return ionc strength for electrolyte
# in same units as given in concM
def getIonicStrength( concM, ionCharge ):
    ionicStrength = 0
    for i in range(0,len(concM)):
        ionicStrength += concM[i] * ionCharge[i]**2
    ionicStrength /= 2.0
        
    return ionicStrength


# return Debye length in nm for electrolyte with given concentration (mol/L)
#
# uses EPS_0 if set, otherwise uses water_EPS_0_25dC
#
# Assumes 25°C (298.15) if temperature(K) is not set
def getDebyeLength( concM, ionCharge ):
    ionicStrength = getIonicStrength( concM, ionCharge )
    
    invDebye = 2.0*ionicStrength
    
    # currently in charge^2 mol/L, convert to SI units
    invDebye *= ( sc.e*sc.e          # charge-squared to C^2
                  * sc.Avogadro      # number of particles / mol
                  * 1000.0 )         # L / m^3
    
    # include dielectric
    
    try:
        myEps = EPS_0
    except:
        myEps = water_EPS_0_25dC
    invDebye /=  myEps * sc.epsilon_0
    
    # thermal energy
    myT = 298.15
    try:
        myT = temperature
    except:
        myT = 298.15 # default 25°C'''
    invDebye /= sc.Boltzmann * myT
    
    # finally, the inverse length itself in 1/m
    invDebye = numpy.sqrt(invDebye)
    
    # return the Debye length in nm
    debyeLength = 1.0e9/invDebye
    
    return debyeLength

# calculate the surface charge corresponding to a given boundary electric field (scaledDfield)
# scaledDfield is in mV/Ang
# surface charge returned here in SI units C/m^2
def getSurfaceChargeSIFromScaledDfield( scaledDfield ):
    try:
        myEps = EPS_0
    except:
        myEps = water_EPS_0_25dC

    DfieldSI = ( sc.epsilon_0 * myEps * scaledDfield
                 / 1000    # V/mV
                 * 1e10    # Ang/m
                 )

    return DfieldSI   # for the simple flat case, the D field is the surface charge

# calculate the scaledDfield corresponding to a given surface charge (SI)
# surface charge is in SI units C/m^2
# scaledDfield return in mV/Ang
def getScaledDfieldFromSurfaceChargeSI( surfaceChargeSI ):
    try:
        myEps = EPS_0
    except:
        myEps = water_EPS_0_25dC

    scaledDfield = surfaceChargeSI / ( sc.epsilon_0 * myEps
                 / 1000    # V/mV
                 * 1e10    # Ang/m
                 )

    return scaledDfield

# return the area per charge for a given surface charge density
# Takes surfaceChargeSI in C/m^2
# Returns area per charge in nm^2
def areaNMSQPerSurfaceChargeSI( surfaceChargeSI ):
    return 1e18 * sc.e / abs(surfaceChargeSI)

# return the area per charge for a given scaledDfield (mV/Ang)
# Returns area per charge in nm^2
def areaNMSQPerChargeFromScaledDfield( scaledDfield ):
    return areaNMSQPerSurfaceChargeSI(getSurfaceChargeSIFromScaledDfield(scaledDfield))
