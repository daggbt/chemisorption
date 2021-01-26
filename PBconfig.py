# structures for setting up a PB calculation
# i.e. setting up electrolyte and boundary conditions
#
# These are conceptual (physical) boundary conditions
# not technical FEniCS/dolfin boundary conditions

import PButils
import numpy

class Ion:
    name=""
    charge=""
    gaussianRadiusAng=0
    concentration=0
    dispersionB=[]
    vacuumDispersionB=[]
    
class Electrolyte:
    ionList = []  # intended as a collection of Ions

class BoundaryCondition:
    type = None  # PButil.BoundaryConditionType
    value = 0  # e.g. the potential (mV), surface charge (C/m^2), or ChargeRegulatedSite
    domainIndex = 0  # index to boundary subdomain

chemisorptionRadiusMultiplier = 2.0  # multiplier of ion radius, over which chemisorbed ion charge is spread. 1 = spread charge over 1 radius thickness, 2 = spread over one diameter
physisorptionOffset = 0.01  # avoid oscillatory noise right on the outerHelmholtzPlane or Distance of Closest Approach, used with ChargeRegulatingIon.physisorptionDistance

class ChargeRegulatingIon:
    electrolyteIndex = 0  # index to ion in electrolyte ion list. Negative value means implicit charge ("background H+")
    pKdissociation = -numpy.inf  # by default, no chemisorption
    physisorptionDistance = 0  # distance from ion to actual surface (accounting for Distance of Closest Approach or Helmholtz layer)
    chemisorptionThickness = 0 # "layer thickness" of chemisorbed ion when treated as volume density (i.e. smearing bound charge throughout this volume)

    
class ChargeRegulatedSite:
    dissociatedCharge = 0
    siteDensityPerMSq = 0
    nativeIonCompetitorIndex = 0  # index to native ion in ionCompetitors list, default to first entry
    ionCompetitors = []  # must always have at least one entry, even if it has electrolyteIndex<0
    pKdoubleBoundDissociation=[]
    implicitNativeIonCharge = 1  # only relevant if native electrolyteIndex < -1. Default "H+"
    implicitNativeIonBulkConcentration = 0  # only relevant if native electrolyteIndex < -1
    implicitNativeIonDispersionB = None  # only relevant if native electrolyteIndex < -1
    implicitNativeNES = None  # only relevant if native electrolyteIndex < -1
    # reference bulk concentrations used to evaluate charge regulation.
    # Differs from bulkConcentration under nonequilibrium conditions.
    # If None, then use current actual bulk concentration (opts.bulkConcentration)
    bulkConcentrationSurfaceReference = None 

class PBconfig:
    electrolyte = Electrolyte()
    boundaryConditionList = []
    # error spotted in J. Chem. Phys. 142, 134707 (2015)
    # eq.12-13,36-38 should add K_{nm},\beta_{ji},K_{ji} alongside K_{mn},\beta_{ij},K_{ij} to Γ_i and B_k
    # Now fixed. Set useIncorrectDoubleBinding=True to recover old erroneous calculation.
    useIncorrectDoubleBinding=False

