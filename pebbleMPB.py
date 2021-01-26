# -*- coding: utf-8 -*-

# spherical symmetry
# calculating an aqueous layer between 2 concentric spherical boundaries

# specifically, reproduce lysozyme calculations
# anu/nonlocal/numerics/DLVO/proteins/lysozyme/salts/hydNaCl-pH12.4-1M/

# or BSA
# anu/nonlocal/numerics/DLVO/proteins/BSA/TanfordBulkToSurf-NaCl-noDCA-pH

# here "polyhistidine", taking advantage of the amine pKa near 7

from dolfin import *
import ufl
import numpy
from scipy import optimize
import scipy  # for scipy.log, without hiding as just "log"
import scipy.constants as sc
import string
import time
import matplotlib.pyplot as plt
#from petsc4py import PETSc  # needed to handle access MPI_Comm_rank() before MPI_FINALIZE
from mpi4py import MPI as MPI4
import h5py
import mpi4py
import os
from PButils import *  # pKwaterDiss etc
import PBconfig
import copy
import numpy as np
import pylab as pltt

NaIon = PBconfig.Ion()
NaIon.name="Na"
NaIon.charge=1
NaIon.gaussianRadiusAng=0.607246  # Na+ bare

hydNaw3Ion = PBconfig.Ion()
hydNaw3Ion.name="hydNaw3"
hydNaw3Ion.charge=1
hydNaw3Ion.gaussianRadiusAng=2.24981  # hydNaw3

dihydrogenphosphateIon = PBconfig.Ion()
dihydrogenphosphateIon.name="dihydrogenphosphate"
dihydrogenphosphateIon.charge=-1
dihydrogenphosphateIon.gaussianRadiusAng=2.2646  # dihydrogenphosphate

hydrogenphosphateIon = PBconfig.Ion()
hydrogenphosphateIon.name="hydrogenphosphate"
hydrogenphosphateIon.charge=-2
hydrogenphosphateIon.gaussianRadiusAng=2.32185  # hydrogenphosphate


trisHIon = PBconfig.Ion()
trisHIon.name="trisH"
trisHIon.charge=1
trisHIon.gaussianRadiusAng=2.67055  # trisH

trisIon = PBconfig.Ion()
trisIon.name="tris"
trisIon.charge=0
trisIon.gaussianRadiusAng=2.69921  # tris

citrateIon = PBconfig.Ion()
citrateIon.name="citrate"
citrateIon.charge=-3
citrateIon.gaussianRadiusAng=3.01718  # citrate

hydrogencitrateIon = PBconfig.Ion()
hydrogencitrateIon.name="hydrogencitrate"
hydrogencitrateIon.charge=-2
hydrogencitrateIon.gaussianRadiusAng=2.97717  # hydrogencitrate


aceticAcid = PBconfig.Ion()
aceticAcid.name="aceticAcid"
aceticAcid.charge=0
aceticAcid.gaussianRadiusAng=2.33034  # aceticAcid

acetateIon = PBconfig.Ion()
acetateIon.name="acetate"
acetateIon.charge=-1
acetateIon.gaussianRadiusAng=2.19534  # acetate


hydroniumIon = PBconfig.Ion()
hydroniumIon.name="hydronium"
hydroniumIon.charge=1
hydroniumIon.gaussianRadiusAng=0.973925  # hydronium

simpleOHion = PBconfig.Ion()
simpleOHion.name="OH"
simpleOHion.charge=-1
simpleOHion.gaussianRadiusAng=1.25953  # simple OH-

hydOHw3Ion = PBconfig.Ion()
hydOHw3Ion.name="hydOHw3"
hydOHw3Ion.charge=-1
hydOHw3Ion.gaussianRadiusAng=2.39243  # hydOHw3

ClIon = PBconfig.Ion()
ClIon.name="Cl"
ClIon.charge=-1
ClIon.gaussianRadiusAng=1.86058  # Cl[alt]

nitrateIon = PBconfig.Ion()
nitrateIon.name="nitrate"
nitrateIon.charge=-1
nitrateIon.gaussianRadiusAng=2.01228  # nitrate

########################################################################################Added from Drew's paper########################################################################################
noDispLiIon = PBconfig.Ion()
noDispLiIon.name="noDispLi+"
noDispLiIon.charge=1
noDispLiIon.gaussianRadiusAng=0.38467

liIon = PBconfig.Ion()
liIon.name="Li"
liIon.charge=1
liIon.gaussianRadiusAng=0.38467

pf6Ion = PBconfig.Ion()
pf6Ion.name="PF_6"
pf6Ion.charge=-1
pf6Ion.gaussianRadiusAng=2.31

bf4Ion = PBconfig.Ion()
bf4Ion.name="BF_4"
bf4Ion.charge=-1
bf4Ion.gaussianRadiusAng=2.09

clo4Ion = PBconfig.Ion()
clo4Ion.name="ClO_4"
clo4Ion.charge=-1
clo4Ion.gaussianRadiusAng=2.17

bro4Ion = PBconfig.Ion()
bro4Ion.name="BrO_4"
bro4Ion.charge=-1
bro4Ion.gaussianRadiusAng=2.27

io4Ion = PBconfig.Ion()
io4Ion.name="IO_4"
io4Ion.charge=-1
io4Ion.gaussianRadiusAng=2.36


####### radius measured using Avogadro software##########
bistriflimideIon = PBconfig.Ion()
bistriflimideIon.name="Bistriflimide"
bistriflimideIon.charge=-1
bistriflimideIon.gaussianRadiusAng=3.42

####### radius measured using Avogadro software##########
emtIon = PBconfig.Ion()
emtIon.name="EM"
emtIon.charge=1
emtIon.gaussianRadiusAng=3.47

####### largest size limit constrained by concentration of 1.2M ##########
XIon = PBconfig.Ion()
XIon.name="X"
XIon.charge=-1
XIon.gaussianRadiusAng=6.287

########################################################################################Added from Drew's paper########################################################################################

hydrogenIon = hydroniumIon   # H3O+
#hydrogenIon = bareHIon      # H+
#hydroxideIon = hydOHw3Ion    # [OH.3H2O]-
hydroxideIon = simpleOHion        # OH-

NaIonSolute = NaIon
#NaIonSolute = hydNaw3Ion

HionElectrolyteIndex = 2
citrateElectrolyteIndex = -1
acetateElectrolyteIndex = 1
temperature=298.15  # water 25°C

class PebbleOptions:
  innerRadius = -1
  R_add_MSA_Ang = 0.54     # 0.54 is for water, check for PC!! R_add is the added radius to calculate the cavity radius
  surfaceTension = 0#41.1 #Propylene Carbonate - air surface tension in Newton per meter. #9.984443793513054* sc.Boltzmann* 1E18*temperature # 41.1 mN/m (=41.1*1E-12 Newtons per nm)
  contactAngle = 31.4 # https://aip.scitation.org/doi/pdf/10.1063/1.4794792?class=pdf
  #interfaceTension = 0.15346905760218352 * sc.Boltzmann*temperature * 1E20 # Calculated from Hamaker constant = 31.50284882429552 (Graphite-Air-PC) and D_0 =1.65 Ang from Israelachvili's book
  solventRadius = 0.275 # in nm. Corrected from the book 'The properties of Solvents' by Yizhak Marcus
  symmetricBoundary = False # True = same material at each boundary. False = distinct different boundaries
  DerjaguinApproximation = 2 #  multiplier for Derjaguin approximation: 1=sphere-sphere, 2 = sphere-plate or cylinder-cylinder
  #pureWater=True  # see backgroundSaltConc instead
  explicitH=False
  #pH = 5.6   # 5.6 is "natural" pH in equilibrium with atmospheric CO2
  pH = 0  # set below
  #pH = 2.0
  pHCounterIonList = [ NaIonSolute, ClIon ]  # ignored if bufferIonList is provide
  naturalpH=7  # could be 5.6 if dissolved CO2 is considered
  #backgroundIonList=[NaIonSolute, ClIon ]
  ########################################################################################Added from Drew's paper########################################################################################
  backgroundIonList=[ ]#liIon, pf6Ion ]
  ########################################################################################Added from Drew's paper########################################################################################
  backgroundIonConcentration = 1.2
  bufferIonList=[]
  ionCharge=[]
  gaussianRadius=[]
  hydgaussianRadius=[]
  stericRadius=[]
  stericEnergyFloor=[]
  ##HamakerConstant_kT = 0
  HamakerConstant_kT = 27.27302327817084; # kT  graphite-water-graphite (Dagastine, isotropic average of graphite
  pureWater = False
  backgroundSaltConc = 0.5   # base concentration in mol/L############################################################################################################
  bulkConcentration=[]    # concentrations of [H+, OH-[, cation, anion]]
  bindingSites=[]
  competitiveBinding = True
  leftPotential=10.0   # mV
  potentialDifference = 100
  boundary=[]
  lengthScale = 1
  potentialScale = 1
  concentrationScale = 1
  ionNESinteraction = []
  useIncorrectDoubleBinding=False
  addIncorrectChemNESenergy=False
  electrostaticOnly = False
  useCavityEnergy = True
  useInterfaceEnergy = True
  ionDCAenergy = []
  ionCavityEnergy = []
  ionInterfaceEnergy = []
  useDCA = False
  cavityRadius = []
  #useDCA = [True, False ]
  printAllSolutions = True
  outerHelmholtzPlane=0  # Helmholtz layer (shift in bulk dielectric constant)
  
opts=PebbleOptions()
pbconfig = PBconfig.PBconfig()


if opts.symmetricBoundary:
  NaIon.dispersionB = 3.478688070139805e-01  # Tavares protein, Na+
  hydNaw3Ion.dispersionB = 1.738694848612457  # Tavares protein, hydNaw3
  dihydrogenphosphateIon.dispersionB = -2.556830765120765   # Tavares protein, dihydrogenphosphate
  hydrogenphosphateIon.dispersionB = -5.270232265785936   # Tavares protein, hydrogenphosphate
  trisHIon.dispersionB = -39.56774275   # Tavares protein, trisH
  trisIon.dispersionB = -5.832406987   # Tavares protein, tris
  citrateIon.dispersionB =-14.95270719    # Tavares protein, citrate
  hydrogencitrateIon.dispersionB = -12.15204999   # Tavares protein, hydrogencitrate
  aceticAcid.dispersionB = -2.234650518378986   # Tavares protein, aceticAcid
  acetateIon.dispersionB = -2.889011084829870   # Tavares protein, acetate
  hydroniumIon.dispersionB = 0.4966042802   # Tavares protein, hydronium
  simpleOHion.dispersionB = -3.857365441437579   # Tavares protein, OH-
  hydOHw3Ion.dispersionB = -1.620644411   # Tavares protein, hydOHw3
  nitrateIon.dispersionB = -1.327143705948489   # Tavares protein, nitrate
  ########################################################################################Added from Drew's paper########################################################################################
  liIon.dispersionB = 1.1446
  pf6Ion.dispersionB = 13.048
  bf4Ion.dispersionB = 13.48
  clo4Ion.dispersionB = 4.5479
  bro4Ion.dispersionB = 2.9088
  io4Ion.dispersionB = -0.90223
  ClIon.dispersionB = 3.5858
  bistriflimideIon.dispersionB = 0
  emtIon.dispersionB = 0
  XIon.dispersionB = 0
  liIon.vacuumDispersionB = -0.5791161484
  pf6Ion.vacuumDispersionB = -67.62159333
  bf4Ion.vacuumDispersionB = -46.83269312
  clo4Ion.vacuumDispersionB = -70.07374511
  bro4Ion.vacuumDispersionB = -80.88967105
  io4Ion.vacuumDispersionB = -93.47613813
  ClIon.vacuumDispersionB = -44.73831769
  bistriflimideIon.vacuumDispersionB = 0
  emtIon.vacuumDispersionB = 0
  XIon.vacuumDispersionB = 0
#######################################################################################Added from Drew's paper########################################################################################	
else:
  NaIon.dispersionB = [3.478688070139805e-01, 3.478688070139805e-01]   # Tavares protein, Na+
  hydNaw3Ion.dispersionB = [1.738694848612457,1.738694848612457]   # Tavares protein, hydNaw3
  dihydrogenphosphateIon.dispersionB = [-2.556830765120765,0]   # Tavares protein, dihydrogenphosphate
  hydrogenphosphateIon.dispersionB = [-5.270232265785936,0]   # Tavares protein, hydrogenphosphate
  trisHIon.dispersionB = [-39.56774275,0]   # Tavares protein, trisH
  trisIon.dispersionB = [-5.832406987,0]   # Tavares protein, tris
  citrateIon.dispersionB = [-14.95270719,0]   # Tavares protein, citrate
  hydrogencitrateIon.dispersionB = [-12.15204999,0]   # Tavares protein, hydrogencitrate
  aceticAcid.dispersionB = [-2.234650518378986,0]   # Tavares protein, aceticAcid
  acetateIon.dispersionB = [-2.889011084829870,0]   # Tavares protein, acetate
  hydroniumIon.dispersionB = [-0.4966042802,0]   # Tavares protein, hydronium
  simpleOHion.dispersionB = [-3.857365441437579,0]   # Tavares protein, OH-
  hydOHw3Ion.dispersionB = [-1.620644411,0]   # Tavares protein, hydOHw3
  nitrateIon.dispersionB = [-1.327143705948489,0]   # Tavares protein, nitrate
  ########################################################################################Added from Drew's paper########################################################################################
  liIon.dispersionB = [1.1446, 1.1446]
  pf6Ion.dispersionB = [13.048, 13.048]
  bf4Ion.dispersionB = [13.48, 13.48]
  clo4Ion.dispersionB = [4.5479, 4.5479]
  bro4Ion.dispersionB = [2.9088, 2.9088]
  io4Ion.dispersionB = [-0.90223, -0.90223]
  ClIon.dispersionB = [3.5858,3.5858]
  bistriflimideIon.dispersionB = [0,0]
  emtIon.dispersionB = [0,0]
  XIon.dispersionB = [0,0]
  liIon.vacuumDispersionB = [-0.5791161484,-0.5791161484]
  pf6Ion.vacuumDispersionB = [-67.62159333,-67.62159333]
  bf4Ion.vacuumDispersionB = [-46.83269312,-46.83269312]
  clo4Ion.vacuumDispersionB = [-70.07374511,-70.07374511]
  bro4Ion.vacuumDispersionB = [-80.88967105,-80.88967105]
  io4Ion.vacuumDispersionB = [-93.47613813,-93.47613813]
  ClIon.vacuumDispersionB = [-44.73831769, -44.73831769 ]   
  bistriflimideIon.vacuumDispersionB = [0,0]
  emtIon.vacuumDispersionB = [0,0]
  XIon.vacuumDispersionB = [0,0]
########################################################################################Added from Drew's paper########################################################################################	
V1 =[]
u_all=False
uIsInitialised=False
previous_u_allf=False


rootMPI=0
#mpi_comm=mpi_comm_world()
mpi_comm=MPI.comm_world

startTime = time.perf_counter()
lastTime=startTime

# needed for parallel handling of mesh? cf. https://fenicsproject.org/qa/7135/parallel-solve-fails/
parameters["ghost_mode"] = "shared_facet"

# fix adaptive mesh
# (see https://fenicsproject.org/qa/6719/using-adapt-on-a-meshfunction-looking-for-a-working-example)
#parameters["refinement_algorithm"] = "plaza_with_parent_facets"

qe =  sc.e  # 1.602176462e-19 C
EPS_VAC = sc.epsilon_0  # 8.854187817E-12 SI units F/m = A^2s^4/(kgm^3) = C^2/(Nm^2) = J/(m V^2) = C/(m V) = C^2 /(m J)
kB = sc.Boltzmann  # 1.3806503e-23 J/K
Avo_N = sc.Avogadro  # 6.02214199e23 atoms/mole
Navo = Avo_N
unity = 1
 
#temperature=273.15   # ice-water, take 0°C
EPS_0 = 66.14 # propylene carbonate at 25°C
#if temperature > (5+273.15):
  #EPS_0 = 78.36   # at 25°C
  #EPS_0 = 66.14   # propylene carbonate at 25°C
#else:
  #EPS_0 = 88.2   # at 0°C, cf. Elbaum-Schick

kT_kJmol = kB*temperature * Avo_N / 1000;   # kT in kJ/mol


# conversion factor for energy of unit charge in potential (mV) to kJ/mol
electrostaticEnergyFactor = (qe * Navo   # C / mol
                             / 1000.0  # V/mV
                             / 1000.0)  # kJ/J

# electrostatic energy (kJ/mol) of given charge in given potential (mV)
if (MPI.rank(mpi_comm) == rootMPI):
	print('Dielectric is {}'.format(EPS_0))
	print('Hamaker Constant is {}'.format(opts.HamakerConstant_kT))
	
####################################################################################################################################################################################################################################
def getBoundaryCondition(leftPotential,potentialDifference,symmetricBoundary):
	opts.leftPotential = leftPotential
	opts.potentialDifference = potentialDifference
	opts.symmetricBoundary = symmetricBoundary
	
	boundaryInner = PBconfig.BoundaryCondition()
	boundaryInner.type = BoundaryConditionType.CONSTANT_POTENTIAL
	boundaryInner.value = opts.leftPotential
	# air-water interface at outer boundary, take as zero charge
	boundaryOuter = PBconfig.BoundaryCondition()
	#boundaryOuter.type = BoundaryConditionType.CHARGE_REGULATED
	boundaryOuter.type = BoundaryConditionType.CONSTANT_POTENTIAL
	if opts.symmetricBoundary:
		boundaryOuter.value = opts.leftPotential
	else:
		boundaryOuter.value = opts.leftPotential - opts.potentialDifference 
	boundaryInner.domainIndex = 0
	if opts.symmetricBoundary:
		pbconfig.boundaryConditionList = [ boundaryInner ]
	else:
		boundaryOuter.domainIndex = 1
		pbconfig.boundaryConditionList = [ boundaryInner, boundaryOuter ]
		
	################### pbconfig.boundaryConditionList = [ boundaryGalena, boundaryBulkLiquid ]
	#if (MPI.rank(mpi_comm) == rootMPI):
		#print('boundaryInner is{} and boundaryOuter is {}'.format(boundaryInner.value, boundaryOuter.value))

####################################################################################################################################################################################################################################	

####################################################################################################################################################################################################################################
def getElectrostaticInteraction( charge, potential ):# charge is not defined. Should it be qe?
  return ( charge * potential 
           * electrostaticEnergyFactor )
####################################################################################################################################################################################################################################
#def singleDispersionEnergy(x,B):
#  return B*exp(-x)

# The steric energy floor is determined by the concentration cap for the ion.
# If the total interaction energy exceeds (is more negative than) this floor
# then the steric energy is applied as the energy difference required to maintain the concentration cap
def getStericEnergyFloor():
  ionVolume=pi*sqrt(pi)*(opts.stericRadius**3)   # nm^3
  concentrationCap = (1.0/ionVolume   # ion/nm^3
                      * 1.0e27        # nm^3/m^3
                      / 1000.0        # m^3/L
                      / Navo)         # mol/ion
  energyFloor = -kT_kJmol * numpy.log( concentrationCap / opts.bulkConcentration )
  return energyFloor

# ~ def getStericEnergy(potential, nes, ionCharge, stericEnergyFloor, ionDCAEnergy=0 , ionCavityEnergy = 0, ionInterfaceEnergy = 0, V=None,debug=False):
def getStericEnergy(potential, nes, ionCharge, stericEnergyFloor, ionCavityEnergy, ionInterfaceEnergy, ionDCAEnergy=0 , V=None,debug=False):
  electrostaticEnergy = getElectrostaticInteraction( ionCharge, potential )
  stericExcess = stericEnergyFloor - (electrostaticEnergy + nes + ionDCAEnergy + ionCavityEnergy + ionInterfaceEnergy)
  #print("getStericEnergy:\n")
  #print stericEnergyFloor
  #print electrostaticEnergy
  #print nes
  #print stericExcess
  #stericEnergy = conditional(gt(stericExcess,0),stericExcess,0)  # i.e. max(0,stericExcess)
  zeroFunction=Constant(0)
  stericEnergy = ufl.Max(stericExcess,zeroFunction)   # this form does some undesirable interpolation, converting [2.36329316, -2.25503678] to [1.75905788, -0.23566907] instead of [2.36329316, 0]
  #stericEnergy = Expression("excess(x[0])>0?excess(x[0]):0",excess=stericExcess)     # nope
  #stericExcessF = project(stericExcess,V)#   ... why does this fail? it works in debug mode below
  # fails first time when potential is "v_1" not "f_9" i.e. TrialFunction not Function
  # error:
  # ...
  #   File "/usr/lib/python2.7/dist-packages/ufl/algorithms/check_arities.py", line 40, in sum
  #  raise ArityMismatch("Adding expressions with non-matching form arguments {0} vs {1}.".format(a, b))
  # ufl.algorithms.check_arities.ArityMismatch: Adding expressions with non-matching form arguments () vs (Argument(FiniteElement('Lagrange', Domain(Cell('interval', 1), label='dolfin_mesh_with_id_4', data='<data with id 4>'), 1, quad_scheme=None), 1, None),).
  #stericEnergy = Function(stericExcessF) 
  # Using stericExcess direct (with TrialFunction potential), gives error:
  #  File "/usr/lib/python2.7/dist-packages/dolfin/functions/function.py", line 287, in __init__
  #  raise TypeError("expected a FunctionSpace or a Function as argument 1")
  #TypeError: expected a FunctionSpace or a Function as argument 1
  #stericEnergy.vector()[stericExcess.vector()<0] = 0.0

  if(debug):
    print("getStericEnergy:\n")
    elV=project(electrostaticEnergy,V)
    nesV=project(nes,V)
    sterExV=project(stericExcess,V)
    sterV=project(stericEnergy,V)
    print("ster floor="+str(stericEnergyFloor))
    print("el",elV.vector().array())
    print("nes",nesV.vector().array())
    print("sterEx",sterExV.vector().array())
    print("ster",sterV.vector().array())
  return stericEnergy

def getHardSphereRadiusFromGaussian( gaussianRadius ):
  #return gaussianRadius * numpy.cbrt( 3.0*numpy.sqrt(numpy.pi)/4. )
  # where did cbrt disappear to???
  return gaussianRadius * numpy.power( 3.0*numpy.sqrt(numpy.pi)/4., 1.0/3.0 )  

def getHardSphereRadiusForCavity( gaussianRadius ):
  #return gaussianRadius * numpy.cbrt( 3.0*numpy.sqrt(numpy.pi)/4. )
  # where did cbrt disappear to???
  return [gaussianRadius * numpy.power( 3.0*numpy.sqrt(numpy.pi)/4., 1.0/3.0 ) for gaussianRadius in gaussianRadius  ]

#return hard wall energy for distance of closest approach
def DCAHardWallEnergy(x,gaussianRadius):
  energy = x*0.0
  hardRadius = getHardSphereRadiusFromGaussian(gaussianRadius)
  if (x<hardRadius):
    #energy = numpy.inf
    energy = 999999  # inf currently causes nan in the solution (when NES is nonzero), use "large" finite workaround
  return energy

# take distance of closest approach against surface on both sides
def twoDCAHardWallEnergy(x,gaussianRadius):
  global separation
  return DCAHardWallEnergy(x,gaussianRadius) + DCAHardWallEnergy(separation-x,gaussianRadius)

# useDCA may be a boolean list indicating whether DCA applies to each surface
class DCAEnergy(UserExpression):
  def __init__(self, gaussianRadius, useDCA, degree=2, element=None, **kwargs):
    self.gaussianRadius = gaussianRadius
    self.useDCA = useDCA
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, xScaled):
    global separation
    x = xScaled * opts.lengthScale
    distanceToSurface = [ x[0], separation-x[0] ]
    values[0] = 0
    for i in range(len(distanceToSurface)):
      if self.useDCA[i]:
        values[0] += DCAHardWallEnergy(distanceToSurface[i],self.gaussianRadius)

  def value_shape(self):
    return ()



def singleCavityEnergy(x,cavityRadius):
  global separation
  conversionUnits = (1.0E-3 # N/mN
					*1.0E-18 # m^2/nm^2
					* Navo # molecules/mol
					/ 1000.0 # J/KJ
					)
  ionRadius_s = cavityRadius + opts.solventRadius #solventRadius calculated from Drew's calculation of Radius_s(= 2.05). Might need correction or reference!!
  hardSphereRadius = cavityRadius - opts.R_add_MSA_Ang/10   
  boundaryForCavity = cavityRadius + 2 * opts.solventRadius
  
  # ~ if (x <= hardSphereRadius):
	  # ~ capHeight = boundaryForCavity - hardSphereRadius
	  
  if (x <= boundaryForCavity):# and x > hardSphereRadius):
	  capHeight = boundaryForCavity - x
	  	  
  else:
	  capHeight = 0
	  	  
  #changeInIonCavityArea = -2 * sc.pi * cavityRadius * capHeight  # No Solvent Accessible Surface Area
  #changeInIonCavityArea = -2 * sc.pi * ionRadius_s * capHeight  # Tim's Area
  changeInIonCavityArea = -2 * sc.pi * (cavityRadius**2 / ionRadius_s )* capHeight   # The one we correct
  	  
  ionCavEnergy = opts.surfaceTension * changeInIonCavityArea * conversionUnits
  return ionCavEnergy
	

class TwoSurfaceCavityEnergy(UserExpression):
  def __init__(self, cavityRadius, useCavityEnergy, degree=2, element=None, **kwargs):
    self.cavityRadius = cavityRadius
    self.useCavityEnergy = useCavityEnergy
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, xScaled):
    global separation
    x = xScaled * opts.lengthScale
    values[0] = singleCavityEnergy(x[0],self.cavityRadius)
    distanceToSurface = [ x[0], separation-x[0] ]
    values[0] = 0
    for i in range(len(distanceToSurface)):
      values[0] += singleCavityEnergy(distanceToSurface[i],self.cavityRadius)

  def value_shape(self):
    return ()


class OneSurfaceCavityEnergy(UserExpression):
  def __init__(self, cavityRadius, useCavityEnergy, degree=2, element=None, **kwargs):
    self.cavityRadius = cavityRadius
    self.useCavityEnergy = useCavityEnergy
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, x):
    values[0] = singleCavityEnergy(x[0],self.cavityRadius)
    #values[0] = singleDispersionEnergy(x[0],self.B, self.gaussianRadius) +  DCAHardWallEnergy(x[0],self.gaussianRadius)
  def value_shape(self):
    return ()

def singleInterfaceEnergy(x,cavityRadius):
	global separation
  
	conversionUnits = (1.0E-3 # N/mN
					*1.0E-18 # m^2/nm^2
					* Navo # molecules/mol
					/ 1000.0 # J/KJ
					)
  
	ionRadius_s = cavityRadius + opts.solventRadius #solventRadius calculated from Drew's calculation of Radius_s(= 2.05). Might need correction or reference!!
     
	boundaryForCavity = cavityRadius + 2 * opts.solventRadius
	hardSphereRadius = cavityRadius - opts.R_add_MSA_Ang/10
	
	# ~ if (x <= hardSphereRadius):
		# ~ changeInIonInterfaceArea = - sc.pi * (ionRadius_s**2 - (hardSphereRadius - opts.solventRadius)**2)
		# ~ # changeInIonInterfaceArea = - sc.pi * (cavityRadius**2 - (hardSphereRadius - 2*solventRadius)**2) # No Solvent Accessible Surface Area
	if (x <= boundaryForCavity):# and x > hardSphereRadius):
		changeInIonInterfaceArea = - sc.pi * (ionRadius_s**2 - (x - opts.solventRadius)**2)
		#changeInIonInterfaceArea = - sc.pi * (cavityRadius**2 - (x - 2*solventRadius)**2) # No Solvent Accessible Surface Area
	else:
		changeInIonInterfaceArea = 0
		
	ionIntEnergy = - opts.surfaceTension * np.cos(opts.contactAngle) * changeInIonInterfaceArea * conversionUnits # Solvent Accessible Surface Area is considered
	#ionIntEnergy = -surfaceTension * np.cos(contactAngle) * changeInIonInterfaceArea * conversionUnits    #No Solvent Accessible Surface Area
  
	return ionIntEnergy
	

class TwoSurfaceInterfaceEnergy(UserExpression):
  def __init__(self, cavityRadius, useInterfaceEnergy, degree=2, element=None, **kwargs):
    self.cavityRadius = cavityRadius
    self.useCavityEnergy = useInterfaceEnergy
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, xScaled):
    global separation
    x = xScaled * opts.lengthScale
    values[0] = singleInterfaceEnergy(x[0],self.cavityRadius)
    distanceToSurface = [ x[0], separation-x[0] ]
    values[0] = 0
    for i in range(len(distanceToSurface)):
      values[0] += singleInterfaceEnergy(distanceToSurface[i],self.cavityRadius)

  def value_shape(self):
    return ()


class OneSurfaceInterfaceEnergy(UserExpression):
  def __init__(self, cavityRadius, useCavityEnergy, degree=2, element=None, **kwargs):
    self.cavityRadius = cavityRadius
    self.useCavityEnergy = useCavityEnergy
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, x):
    values[0] = singleInterfaceEnergy(x[0],self.cavityRadius)
    #values[0] = singleDispersionEnergy(x[0],self.B, self.gaussianRadius) +  DCAHardWallEnergy(x[0],self.gaussianRadius)
  def value_shape(self):
    return () 
    
    
####################################################################################################################################################################################
#Dispersion energy using cavity model and from non gaussian radius. It returns dispersion energy in kJ/mol
def singleNonGaussianDispersionEnergy(x,B,vacB,gaussianRadius):
	hardSphereRadius = getHardSphereRadiusFromGaussian( gaussianRadius )
	cavityRadius = hardSphereRadius + opts.R_add_MSA_Ang/10 
	ionRadius_s = cavityRadius + opts.solventRadius #solventRadius calculated from Drew's calculation of Radius_s(= 2.05). Might need correction or reference!!
     
	boundaryForCavity = cavityRadius + 2 * opts.solventRadius
  # B in 1e-50 J/m^3, return kJ/mol via nm
	units = (1.0E-50 # J/m^3
			*1.0E27 # nm^3/m^3
			* Navo # molecules/mol
			/ 1000.0 # J/KJ
			)
	
	if (x <= hardSphereRadius):
		capHeight = boundaryForCavity - hardSphereRadius
		D = hardSphereRadius
		
	elif (x <= boundaryForCavity and x > hardSphereRadius):
		capHeight = boundaryForCavity - x
		D = x
		
	else:
		capHeight = 0
		D = x
		
	#changeInIonCavityArea =  -2 * sc.pi * cavityRadius * capHeight   # No SASA
	#changeInIonCavityArea = -2 * sc.pi * ionRadius_s * capHeight     # Tim's area
	changeInIonCavityArea = -2 * sc.pi * (cavityRadius**2 / ionRadius_s )* capHeight    # Our corrected area
	
	
	interactionFactorVacuum = abs(changeInIonCavityArea) / (4 * sc.pi * cavityRadius**2)
	interactionFactorSolvent = 1 - interactionFactorVacuum
	dispersionEnergy = units * B/D**3              # No cavity model is assumed
	# ~ dispersionEnergy = units * (( interactionFactorSolvent * B ) + (interactionFactorVacuum * vacB ))/ (D**3)    # Our corrected dispersion
	return dispersionEnergy

##################################################################################################################################################################################################################

### Old Dispersion energy using gaussianRadius and without cavity model
#return dispersion energy in kJ/mol
def singleDispersionEnergy(x,B,gaussianRadius):
  # B in 1e-50 J/m^3, return kJ/mol via nm
  units = 1.0E-50 * Navo *1.0E27/ 1000.0
  r = x/gaussianRadius
  if ( r<3e-6 ):
    g=4.0/3.0
  else:
    g = exp(-r**2)*(1.0-1.0/(2.0*r**2)) + sqrt(pi)*r*erf(r)*(1.0+1.0/(4.0*r**4)) - sqrt(pi)*r
  
  energy = units * 4.0*B * g / (sqrt(pi)*gaussianRadius**3) ################### why gaussianRadius?
  
  return energy


##################################################################################################################################################################################################################


# model nonelectrostatic interaction
# B may be a single value used for all surfaces,
# or an array for each surface in turn
class TwoSurfaceDispersionEnergy(UserExpression):
  def __init__(self, B, vacB, gaussianRadius, degree=2, element=None, **kwargs):
    self.B = B
    self.vacB = vacB
    self.gaussianRadius = gaussianRadius
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, xScaled):
    global separation
    x = xScaled * opts.lengthScale
    distanceToSurface = [ x[0], separation-x[0] ]
    values[0] = 0
    for i in range(len(distanceToSurface)):
      if numpy.size(self.B) == 1:
        B = self.B
        vacB = self.vacB
      else:
        B =  self.B[i]
        vacB = self.vacB[i]
        
      values[0] += singleNonGaussianDispersionEnergy(distanceToSurface[i], B, vacB, self.gaussianRadius)
      #values[0] += singleDispersionEnergy(distanceToSurface[i], B, vacB, self.gaussianRadius)
    #values[0] = singleDispersionEnergy(x[0],self.B, self.gaussianRadius) + singleDispersionEnergy(separation-x[0],self.B, self.gaussianRadius) + twoDCAHardWallEnergy(x[0],self.gaussianRadius)
  def value_shape(self):
    return ()


class OneSurfaceDispersionEnergy(UserExpression):
  def __init__(self, B, vacB, gaussianRadius, degree=2, element=None, **kwargs):
    self.B = B
    self.vacB = B
    self.gaussianRadius = gaussianRadius
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, x):
    #values[0] = singleDispersionEnergy(x[0],self.B, self.vacB, self.gaussianRadius)
    values[0] = singleNonGaussianDispersionEnergy(x[0],self.B, self.vacB, self.gaussianRadius)
    #values[0] = singleDispersionEnergy(x[0],self.B, self.gaussianRadius) +  DCAHardWallEnergy(x[0],self.gaussianRadius)
  def value_shape(self):
    return ()



	
	

def plotEnergy(minX,maxX,cavityRadius,B,vacB,gaussianRadius,ions,debyeLength,leftPotential):
	global separation
	cav = []
	interface = []
	disp = []
	area1 = []
	area2 = []
	factor = 1000 /(sc.Avogadro*sc.Boltzmann*temperature)
	factorDisp = 1000 /(sc.Avogadro*sc.Boltzmann*temperature) # added to switch on and off the dispersion
	ionRadius_s = cavityRadius + opts.solventRadius      
	boundaryForCavity = cavityRadius + 2 * opts.solventRadius
	total = []
	coulombic = []
	NONEScoulombic = []
	# ~ steric = []
	allData = convertXDMF2OctavePB(separation, "-leftPotential{}-salt{}M".format(opts.leftPotential,opts.backgroundSaltConc)) 
	x = [i/10 for i in allData[:,0]]
	potential = [1*i for i in allData[:,1]]
	if (ions == 0):
		Electrostaticfactor = sc.e/(sc.Boltzmann*temperature*1000 )
		coulombic = [Electrostaticfactor*i for i in allData[:,1]]
		# ~ coulombic = [1.6*i/(4.11) for i in allData[:,1]]
		#totalInteractionPotential = [1.6*i/(4.11) for i in allData[:,1]]
		entropic = [0 for i in range(len(x))]	
		# ~ entropic = [np.log(i) for i in allData[:,5]]
		flag = 'Li'
		stericEnergyFloor = opts.stericEnergyFloor[0]
	else:
		Electrostaticfactor = -sc.e/(sc.Boltzmann*temperature*1000 )
		coulombic = [Electrostaticfactor*i for i in allData[:,1]]
		# ~ coulombic = [-1.6*i/(4.11) for i in allData[:,1]]
		#totalInteractionPotential = [-1.6*i/(4.11) for i in allData[:,1]]
		entropic = [np.log(i) for i in allData[:,6]]
		flag = 'Negative'
		stericEnergyFloor = opts.stericEnergyFloor[1]
	
	x = [i/10 for i in allData[:,0]]
	ionCharge = sc.e
	n = len(x) 
	k = int(n/2)
	x = x[0:k]
	i = 0
	# ~ getStericEnergy(potential, nes, ionCharge, stericEnergyFloor, ionCavityEnergy, ionInterfaceEnergy, ionDCAEnergy=0 , V=None,debug=False):		
	#nesEnergy = list( TwoSurfaceDispersionEnergy(B, vacB, gaussianRadius,ion.charge,element=nesEl) for ion in opts.ionList )
	for xvalue in x:
		# ~ pot = potential[i]
		# ~ i = i+1
		cavity = singleCavityEnergy(xvalue,cavityRadius)*factor
		inter = singleInterfaceEnergy(xvalue,cavityRadius)*factor
		dispersion = singleNonGaussianDispersionEnergy(xvalue,B,vacB,gaussianRadius)*factorDisp
		# ~ stericEnergy = getStericEnergy(pot, dispersion, ionCharge, stericEnergyFloor, cavity, interface, ionDCAEnergy=0 , V=None,debug=False)
		totalEnergy = cavity + inter + dispersion # + stericEnergy 
		cav.append(cavity)
		interface.append(inter)
		disp.append(dispersion)
		# ~ steric.append(stericEnergy)
		total.append(totalEnergy)
		# ~ NONEScoulombic.append(Electrostaticfactor * leftPotential * np.exp(-xvalue/debyeLength))# .001 is factor for mV
		
	# ~ total = [total[i] + coulombic[i] for i in range(len(x))]
	total = [total[i] + coulombic[i] + entropic[i] for i in range(len(x))]
	#totalInteractionPotential = totalInteractionPotential[0:k]
	coulombic = coulombic[0:k]
	NONEScoulombic = NONEScoulombic[0:k]
	entropic = entropic[0:k]
	cav = cav[0:k]
	interface = interface[0:k]
	disp = disp[0:k]
	total = total[0:k]
	energyAll = []
	energyLabels = ['Coulombic','Dispersion','Cavity','Interface'] 
	# ~ if (MPI.rank(mpi_comm) == rootMPI):
		# ~ plt.figure()
		# ~ plt.subplot(211)	
		# ~ #plt.plot(x,coulombic,x,cav,x,interface,x,disp)
		# ~ plt.plot(x,disp)#coulombic,x,entropic,x,cav,x,interface,x,disp)
		# ~ plt.legend(('Coulombic','Entropic','Cavity','Interface','Dispersion'), loc = 'lower center')
		# ~ plt.xlabel('x (nm)')
		# ~ plt.ylabel('energy (kT)')
		# ~ plt.title('Energy of a molecule')
		# ~ plt.subplot(212)
		# ~ plt.plot(x,total)
		# ~ plt.xlabel('x (nm)')
		# ~ plt.ylabel('energy (kT)')
		# ~ plt.savefig(flag + '_ion.png')
	
	if (MPI.rank(mpi_comm) == rootMPI):
		fig= plt.figure(figsize=(12,6))
		plt.subplot(231)
		plt.plot(x,coulombic,label='Coulombic')#,x,entropic,x,cav,x,interface,x,disp)
		#plt.xlabel('x (nm)', fontsize = 22.0)
		plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='center', fontsize = 14.0)
		# ~ plt.subplot(232)
		# ~ plt.plot(x,entropic,label='Entropic')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ #plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ #plt.ylabel('energy (kT)', fontsize = 14.0)
		# ~ l = plt.legend(loc='upper right', fontsize = 14.0)
		plt.subplot(232)
		plt.plot(x,disp,label='Dispersion')#,x,entropic,x,cav,x,interface,x,disp)
		#plt.xlabel('x (nm)', fontsize = 14.0)
		#plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='center', fontsize = 14.0)
		plt.subplot(233)
		plt.plot(x,cav,label='Cavity')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='center', fontsize = 14.0)
		plt.subplot(234)
		plt.plot(x,interface,label='Interface')#,x,entropic,x,cav,x,interface,x,disp)
		plt.xlabel('x (nm)', fontsize = 14.0)
		plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='upper right', fontsize = 14.0)
		
		# ~ plt.subplot(235)
		# ~ plt.plot(x,entropic,label='Entropic')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ plt.xlabel('x (nm)', fontsize = 14.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 14.0)
		# ~ l = plt.legend(loc='lower right', fontsize = 14.0)
		
		plt.subplot(235)
		plt.plot(x,total,label='Total')
		plt.xlabel('x (nm)', fontsize = 14.0)
		# ~ #plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='upper right', fontsize = 14.0)
		
		plt.subplot(236)
		plt.plot(x, coulombic,x, disp, x, cav,x, interface)#,x,entropic,x,cav,x,interface,x,disp)
		plt.xlabel('x (nm)', fontsize = 14.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 14.0)
		l = plt.legend(loc='lower right', fontsize = 14.0)
		fig.suptitle(flag + '_ion',y=0.92, fontsize = 14.0)
				
		directory = opts.backgroundIonList[1].name
		try:
			if not os.path.exists(directory):
				os.makedirs(directory)
		except OSError:
			print ('Directory already exists 1 ' +  directory)
		FileName = '{}/leftPot{}-potDiff{}_{}_ion.png'.format(directory,opts.leftPotential, opts.potentialDifference, flag)
		#numpy.savetxt(potentialFileName, allData, '%.10g', '\n', '\t', header=header)
		plt.savefig(FileName)
		
		# ~ fig= plt.figure(figsize=(14,8))
		# ~ plt.subplot(221)
		# ~ plt.plot(x,coulombic,label='Coulombic')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ l = plt.legend(loc='center', fontsize = 20.0)
		# ~ #plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 22.0)
		# ~ plt.subplot(222)
		# ~ plt.plot(x,cav,label='Cavity')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ l = plt.legend(loc='center', fontsize = 20.0)
		# ~ #plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 22.0)
		# ~ plt.subplot(223)
		# ~ plt.plot(x,interface,label='Interface')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ l = plt.legend(loc='center', fontsize = 20.0)
		# ~ plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 22.0)
		# ~ plt.subplot(224)
		# ~ plt.plot(x,disp,label='Dispersion')#,x,entropic,x,cav,x,interface,x,disp)
		# ~ l = plt.legend(loc='center', fontsize = 20.0)
		# ~ plt.xlabel('x (nm)', fontsize = 22.0)
		# ~ plt.ylabel('energy (kT)', fontsize = 22.0)
		# ~ l = plt.legend(loc='center', fontsize = 20.0)
		# ~ fig.suptitle(flag + ' ion',y=0.5, fontsize = 22.0)
		# ~ plt.savefig(flag + '_ion.png', fontsize = 22.0)
		#plt.show()



# concentration cell: different on either side of salt bridge      
class cellConcentration(UserExpression):
  def eval(self, values, x):
    global separation
    if (x[0]<separation/2.0):
      concentration=1.0
    else:
      concentration=0.1
    values[0] = concentration
  def value_shape(self):
    return ()

def DebyeHuckelSingle(x,DebyeLength,surfacePotential):
  return surfacePotential*exp(-x/DebyeLength)


class DebyeHuckelTwoSide(UserExpression):
  def __init__(self, DebyeLength,surfacePotential, degree=2, element=None, **kwargs):
    self.DebyeLength = DebyeLength
    self.surfacePotential = surfacePotential
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)
  
  def eval(self, values, x):
    values[0] = self.surfacePotential*exp(-x[0]/self.DebyeLength) + self.surfacePotential*exp(-x[0]/self.DebyeLength)
  def value_shape(self):
    return ()

    
# control parameters


#surfacePotential=1000
#surfacePotentialRight = 1000.0  # mV
global separation
separation = 10.0



neutralHconc = 10.0**(-neutralpH)
neutralOHconc = 10.0**(neutralpH-pKwaterDiss)


bareLiGaussianRadius = 0.384668  # Li+ bare
hydLiw5GaussianRadius = 2.56195  # hydLiw5
LiGaussianRadius = bareLiGaussianRadius# hydLiw5GaussianRadius

bareNaGaussianRadius = 0.607246  # Na+ bare
hydNaw3GaussianRadius = 2.24981  # hydNaw3
NaGaussianRadius = hydNaw3GaussianRadius

KGaussianRadius = 0.960101     # K+
RbGaussianRadius = 1.12226     # Rb+
CsGaussianRadius = 1.47092     # Cs+

H3OGaussianRadius = 0.973925    # H3O+
hydOHw3GaussianRadius = 2.39243  # hydOHw3
  
ClGaussianRadius = 1.86058    # Cl- alt
ClO4GaussianRadius = 2.17287  # ClO4-
HCO3GaussianRadius = 2.05509    # HCO3-
nitrateGaussianRadius = 2.0123  # NO3-
tungstateGaussianRadius = 2.99996  # WO4--
SCNGaussianRadius = 2.17762  # SCN-


Hcharge = 1
OHcharge = -1
HCO3charge = -1
cationCharge = 1  # K+ etc
anionCharge = -1  # Cl-, nitrate

'''
# order H+, OH-, [ Li+, Na+, tungstate ] , [ background cation, anion ]
def updateElectrolyteManual():
  global opts

  cationCharge = 1  # K+, etc
  anionCharge = -1  # Cl-, nitrate

  if (opts.cation=="Li"):
    cationRadiusAng = LiGaussianRadius
  elif  (opts.cation=="K"):
    cationRadiusAng = KGaussianRadius
  elif  (opts.cation=="Cs"):
    cationRadiusAng = CsGaussianRadius
  elif  (opts.cation=="Rb"):
    cationRadiusAng = RbGaussianRadius
  else: # default hydNaw3
    cationRadiusAng = hydNaw3GaussianRadius
  
  if (opts.anion=="nitrate"):
    anionRadiusAng = nitrateGaussianRadius
  elif (opts.anion=="SCN"):
    anionRadiusAng = SCNGaussianRadius
  else: # default Cl-
    anionRadiusAng = ClGaussianRadius

  
  Hconc = 10.0**(-opts.pH)
  OHconc=10.0**(opts.pH-pKwaterDiss)
  
  acidbaseCationConc=0
  acidbaseAnionConc=0
  if ( opts.pH > 7 ):
    # pH set with cation-OH
    acidbaseCationConc = OHconc - neutralOHconc
  else:
    # pH set with H-anion
    acidbaseAnionConc = Hconc - neutralHconc

  cationConc = opts.backgroundSaltConc+acidbaseCationConc;
  anionConc = opts.backgroundSaltConc+acidbaseAnionConc;
  if opts.explicitH:
    ionConcentrations = [  cationConc, anionConc, Hconc, OHconc ];
  else:
    ionConcentrations = [ opts.backgroundSaltConc, opts.backgroundSaltConc ];
  
  if (MPI.rank(mpi_comm) == rootMPI):
    print ("initial conc=",ionConcentrations)
  
  opts.bulkConcentration=numpy.array(ionConcentrations)

  if opts.explicitH:
    opts.ionCharge=numpy.array([cationCharge, anionCharge,Hcharge, OHcharge  ])
    gaussianRadiusAngstrom=numpy.array([cationRadiusAng, anionRadiusAng, H3OGaussianRadius, hydOHw3GaussianRadius  ])   # H3O+, OH.3H2O
  else:
    opts.ionCharge=numpy.array([cationCharge, anionCharge  ])
    gaussianRadiusAngstrom=numpy.array([cationRadiusAng, anionRadiusAng ])   # H3O+, OH.3H2O
  
  opts.gaussianRadius=gaussianRadiusAngstrom/10.0  # in nm

  if opts.useDCA:
      opts.ionDCAenergy = [ DCAEnergy( opts.gaussianRadius[i], opts.useDCA, degree=1 ) for i in range(len(opts.gaussianRadius)) ]
  else:
      opts.ionDCAenergy = numpy.zeros(len(opts.gaussianRadius))
  opts.stericRadius=opts.gaussianRadius.copy()
  opts.stericEnergyFloor=getStericEnergyFloor()
  
'''
# Find the electroneutrality coefficients (defining electrolyte stoichiometries)
#
def getIonStoichiometries( ionList ):
  # algorithm follows electrolyte.m, electropot.cpp
  stoichiometry = numpy.ones( len(ionList) );
  ionCharge = numpy.array( [ ion.charge for ion in ionList] )
  
  maxStoich=20;

  sIndex = 0;
  minLevel=1;
  searching=True;
  while( searching ):
    chargeBalance = numpy.dot(stoichiometry, ionCharge)
    if numpy.isclose( chargeBalance, 0, atol=1e-10 ):
      searching=False;
      break;

    # increment
    stoichiometry[sIndex] += 1;
    # cascade over back to start
    # (starting from minLevel so as not to repeat states)
    while ( stoichiometry[sIndex] > maxStoich ):
      stoichiometry[sIndex] = minLevel;
      sIndex += 1;
      if (sIndex>=len(stoichiometry)+1):
        # increment minLevel since all lower level permutations
        # will have been tested by now
        minLevel += 1;
        sIndex=1;
      stoichiometry[sIndex]+=1;

    # check if final state has been reached
    searching=False;
    for  s in range(len(stoichiometry)):
      if ( stoichiometry[s] < maxStoich):
        searching=True;
        continue;

  if ( chargeBalance != 0 ):
    
    # If there is only one ion (e.g. counterion to a surface)
    # then by construction the electrolyte is not charge balanced
    # (the system is only electroneutral taking surface+electrolyte together)
    # In this case set the stoichiometry to "1"
    # to use the single ion's given concentration
    
    # For the alternative case of with multiple ions (with user-defined concentrations)
    # the (unbalanced) stoichiometry found at this point is almost certainly not relevant, so
    # do not pass it on.

    if ( len(ionList) == 1 ):
      stoichiometry[0] = 1;
    else:
      # no charge balance found
      print("Charges in bulk solution are not balanced.");

  return stoichiometry
      

# distributes the given ionConcentration across the ionList
# ionConcentration should be provided as a numpy array (it may be scalar with ndim=0, i.e. single value)
# If ionConcentration is a single number (scalar array) then it is treated
# as the concentration of the salt
# indicated by the ionList, and distributed in accordance with electroneutrality
# If ionConcentration is an array then it is assumed to specify concentrations
# for each ion in ionList individually
def setElectrolyteConcentrations( ionList, ionConcentration ):
  if ionConcentration.ndim==0 or len(ionConcentration) == 1:
    # single concentration provided: treat as salt concentration
    # taking ionList as an electroneutral salt
    stoichiometry = getIonStoichiometries( ionList )
    ionConcentrations = stoichiometry * ionConcentration
  else:
    ionConcentrations = ionConcentration
    
  for i in range(len(ionConcentrations)):
    ionList[i].concentration = ionConcentrations[i]
    
  return ionList

# use Henderson-Hasselback to set buffer concentrations
# bufferIonList is presented as [ protonated, deprotonated, counterion ]
def setBufferIonConcentrations( pH, bufferIonList, bufferpKa, bufferConcentration ):
  ratio_deProt_on_Prot = 10**(pH-bufferpKa)
  # r = 10**(pH-bufferpKa) = [A]/[HA] = [A]/([buffer]-[A]), so [A] = [buffer]r/(1+r)
  deprotConc = bufferConcentration * ratio_deProt_on_Prot/(1+ratio_deProt_on_Prot)
  protConc = bufferConcentration - deprotConc
  bufferIonList[0].concentration = protConc
  bufferIonList[1].concentration = deprotConc
  counterConc = -sum((bufferIonList[i].charge*bufferIonList[i].concentration for i in (0,1) ))

  # the amount of dry buffer added to water which is not already at the target pH
  # will have a different ratio of [A]/[HA], expressed via the amount of counterion
  # e.g. if target pH is higher, some A will convert to HA, removing H+.
  # so to achieve electroneutrality at pH away from "natural pH", need extra counterion
  # corresponding to the A which was lost (or vice versa).
  # The amount required is measured by ex[OH-]-ex[H+].
  neutralHconc = 10**(-neutralpH)
  neutralOHconc = 10**(neutralpH-pKwaterDiss)
  Hconc = 10**(-pH)
  OHconc = 10**(pH-pKwaterDiss)
  excessHconc = Hconc-neutralHconc
  excessOHconc = OHconc-neutralOHconc
  counterConc = (counterConc+excessOHconc-excessHconc)/bufferIonList[2].charge
  bufferIonList[2].concentration = counterConc
  
  return bufferIonList
  
# standard electrolyte structure:
# 1) salt: first background salt is added (at natural pH) according to backgroundSaltConc
# 2) pH: either pH is adjusted using bufferIons (if provided), or else with pHCounterIon
def updateElectrolyte():
  global opts

  ionList = []
  if opts.backgroundIonList:
    backgroundIonList = [ copy.copy(ion) for ion in opts.backgroundIonList ]
    backgroundIonList = setElectrolyteConcentrations( backgroundIonList, numpy.array(opts.backgroundIonConcentration) )
    ionList.extend(backgroundIonList)

  if opts.explicitH:
    Hion = copy.copy(hydrogenIon)
    OHion = copy.copy(hydroxideIon)
    Hion.concentration = 10.0**(-opts.pH)
    OHion.concentration = 10.0**(opts.pH-pKwaterDiss)
    HOHions = [ Hion, OHion ]
  else:
    HOHions = None
        
  if opts.bufferIonList:
    # use buffer to balance pH charge
    bufferIonList = [ copy.copy(ion) for ion in opts.bufferIonList ]
    bufferIonList = setBufferIonConcentrations( opts.pH, opts.bufferIonList, opts.bufferpKa, opts.bufferConcentration )
    ionList.extend(bufferIonList)
  else:
    # use pHCounterIonList to balance pH charge
    # (if pH is different from natural pH)
    if not numpy.isclose(opts.pH, opts.naturalpH, atol=1e-10):
      if opts.pH > opts.naturalpH:
        pHCounterIon = copy.copy(opts.pHCounterIonList[0])   # high pH, OH- is balanced by cation
        pHCounterIon.concentration = OHion.concentration - 10.0**(opts.naturalpH-pKw)
      else:
        pHCounterIon = copy.copy(opts.pHCounterIonList[1])   # low pH, H+ is balanced by anion
        pHCounterIon.concentration = Hion.concentration - 10.0**(-pKw)
      ionList.extend(pHCounterIon)

  # place H+/OH- last
  if HOHions:
    ionList.extend(HOHions)
    
  # clean up multiple instances of the same ion in the ionList
  # ions are identified by name, the first instance is kept in place
  ionNames = [ ion.name for ion in ionList ]
  uniqueIonNames = set( ionNames )
  for ionName in uniqueIonNames:
    ionIndexInList = list( i for i, ion in enumerate(ionList) if ion.name==ionName  )
    conc = sum( ion.concentration for ion in ionList if ion.name==ionName )
    ionList[ionIndexInList[0]].concentration = conc
    for n in range(1,len(ionIndexInList)):
      del ionList[ionIndexInList[n]]

  # validate electroneutrality
  totalChargeConcentration = sum((ion.charge*ion.concentration for ion in ionList))
  if not numpy.isclose(totalChargeConcentration, 0, atol=1e-10):
    raise ValueError("Bulk ion concentrations do not provide electroneutrality for pH {}.\n{}".format(opts.pH, list((ion.name,ion.charge,ion.concentration) for ion in ionList)))
 
  opts.ionList=ionList

  # for now pebble uses a separate ionCharge and bulkIonConcentration
  # refactor it out later
  opts.ionCharge = list( ion.charge for ion in opts.ionList )
  opts.bulkConcentration = list( ion.concentration for ion in opts.ionList )
  gaussianRadiusAngstrom = numpy.array( list(ion.gaussianRadiusAng for ion in opts.ionList) )
  opts.gaussianRadius=gaussianRadiusAngstrom/10.0  # in nm

  if not opts.cavityRadius:
	    
	  opts.cavityRadius = [getHardSphereRadiusFromGaussian(opts.gaussianRadius[i]) + opts.R_add_MSA_Ang/10 for i in range(len(opts.gaussianRadius))]
  
  if opts.useDCA:
      opts.ionDCAenergy = [ DCAEnergy( opts.gaussianRadius[i], opts.useDCA, degree=1 ) for i in range(len(opts.gaussianRadius)) ]
      
  else:
      opts.ionDCAenergy = numpy.zeros(len(opts.gaussianRadius))
  opts.stericRadius=opts.gaussianRadius.copy()
  # ~ opts.stericRadius[0]=LiGaussianRadius/10 ###################################### Radius of hydrated Li+
  opts.stericEnergyFloor=getStericEnergyFloor()
  
  if opts.useCavityEnergy:
	  opts.ionCavityEnergy = [TwoSurfaceCavityEnergy( opts.cavityRadius[i], opts.useCavityEnergy, degree = 1) for i in range(len(opts.cavityRadius)) ]
  else: 
	  opts.ionCavityEnergy = numpy.zeros(len(opts.cavityRadius))
  if opts.useInterfaceEnergy:
	  opts.ionInterfaceEnergy = [TwoSurfaceInterfaceEnergy(opts.cavityRadius[i], opts.useInterfaceEnergy, degree = 1) for i in range(len(opts.cavityRadius))]
  else:
	  opts.ionInterfaceEnergy = numpy.zeros(len(opts.cavityRadius))
	  
############################# updateElectrolyte() #################################
  

################################################################################################################################################################################################################
################################################################################################################################################################################################################
HsurfaceB_protein = -0.382159;  # Tavares protein
HsurfaceB=HsurfaceB_protein
#HsurfaceB=0
siteHsurfaceB=HsurfaceB;

boundaryInner = PBconfig.BoundaryCondition()
boundaryInner.type = BoundaryConditionType.CONSTANT_POTENTIAL
boundaryInner.value = opts.leftPotential

# air-water interface at outer boundary, take as zero charge
boundaryOuter = PBconfig.BoundaryCondition()
boundaryOuter.type = BoundaryConditionType.CONSTANT_POTENTIAL
if opts.symmetricBoundary:
  boundaryOuter.value = opts.leftPotential
else:
  boundaryOuter.value = opts.leftPotential - opts.potentialDifference ####################################################

boundaryInner.domainIndex = 0
if opts.symmetricBoundary:
  pbconfig.boundaryConditionList = [ boundaryInner ]
else:
  boundaryOuter.domainIndex = 1
  pbconfig.boundaryConditionList = [ boundaryInner, boundaryOuter ]
  #print('boundaryInner is{} and boundaryOuter is {}'.format(boundaryInner, boundaryOuter))


####################################### pbconfig.boundaryConditionList = [ boundaryGalena, boundaryBulkLiquid ]

# Sub domain for Dirichlet boundary condition
class DirichletBoundaryLeft(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0])<DOLFIN_EPS
  
class DirichletBoundaryRight(SubDomain):
  def inside(self, x, on_boundary):
    global separation
    return on_boundary and abs(x[0]-separation/opts.lengthScale)<DOLFIN_EPS

class DirichletBoundaryRight2(SubDomain):
  def inside(self, x, on_boundary):
    global separation2
    return on_boundary and abs(x[0]-separation2/opts.lengthScale)<DOLFIN_EPS

boundaryTol = 1e-6

class LeftBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0]) < boundaryTol
    
class RightBoundary(SubDomain):
  def __init__(self, separation):
    self.separation = separation/opts.lengthScale
    SubDomain.__init__(self) # Call base class constructor!
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0] - self.separation) < boundaryTol


class LeftBoundary2(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0]) < boundaryTol

class RightBoundary2(SubDomain):
  def __init__(self, separation2):
    self.separation2 = separation2/opts.lengthScale
    SubDomain.__init__(self) # Call base class constructor!
  def inside(self, x, on_boundary):
    return on_boundary and abs(x[0] - self.separation2) < boundaryTol

# use the given referenceBoundary (a member of pbconfig.boundaryConditionList)
# to set the potential scale (in mV)
# If boundary type is CONSTANT_POTENTIAL, then use its value
# If CONSTANT_CHARGE, then estimate the potential scale using the Gouy-Chapman equation
# If CHARGE_REGULATED, then use bulk concentrations
def getReferencePotential( referenceBoundary ):
  refPotential = 0
  debyeLength = getDebyeLength(opts.bulkConcentration,opts.ionCharge)  # represents bulk concentration
  ionicStrength = getIonicStrength(opts.bulkConcentration,opts.ionCharge)  # represents bulk concentration, never mind the inconsistency, it's for scale only
  GouyChapmanCharge = lambda pot: 4*ionicStrength*sc.Avogadro*1000*sc.e*debyeLength*1e-9*numpy.sinh(sc.e*pot/1000/(2*sc.Boltzmann*temperature))
  if referenceBoundary.type == BoundaryConditionType.CONSTANT_POTENTIAL:
    # constant potential given in mV
    refPotential = referenceBoundary.value
    #print('REFPOTENTIAL under if IS {}'.format(refPotential))
  elif referenceBoundary.type == BoundaryConditionType.CONSTANT_CHARGE:
    # Neumann constant charge given in C/m^2
    # use Gouy-Chapman to relate charge to potential, see Hunter, Foundations of Colloid Science Vol 1, p.335 eq.6.3.27
    gc = lambda pot: GouyChapmanCharge(pot) - referenceBoundary.value
    refPotential = scipy.optimize.brentq(gc, -20000, 20000)
    print('REFPOTENTIAL elif IS {}'.format(refPotential))
  elif referenceBoundary.type == BoundaryConditionType.CHARGE_REGULATED:
    # Neumann(Robin) regulated charge in C/m^2
    # given as list of regulated sites
    surfaceChargeSIbulk = ChargeRegulatedSurfaceChargeSI(referenceBoundary.value, numpy.zeros(len(opts.bulkConcentration)), 0 )
    gc = lambda pot: GouyChapmanCharge(pot) - surfaceChargeSIbulk
    refPotential = scipy.optimize.brentq(gc, -20000, 20000)

    print('REFPOTENTIAL IS {}'.format(refPotential))
  return max( 1, abs(refPotential) )
  
# Nonlinear solvent can handle this number in ChargeDistributionFunction,
# but not in this form of explicit variables (which yields a false constant solution).
# So have to collect them.
# Sets Poisson system in mV and nm
PoissonChargeFactor =  ( 1000.0   # mV/V
                         *1000.0   # L/m^3
                         *1.0e-18  # m^2 / nm^2
                         * qe*Navo
                         /(EPS_VAC * EPS_0) )  #  V m / C 

# Define error tolerance
tol = 1.e-3
baseMeshDensity=1000
#baseMeshDensity=500
#mesh = IntervalMesh(meshDensity,0,separation)
#V = FunctionSpace(mesh, "CG", 1)

#... ok pointwise but in print Cu2+ conc exceeds cap
def IonConcentration(concentration,u,nes,ionCharge,stericEnergyFloor,ionDCAEnergy,ionCavityEnergy,ionInterfaceEnergy, V,debug=False):
  stericEnergy=getStericEnergy(u,nes,ionCharge,stericEnergyFloor,ionCavityEnergy,ionInterfaceEnergy, ionDCAEnergy, V,debug)
  ionConcentration = concentration * exp(-getElectrostaticInteraction(ionCharge,u)/kT_kJmol)*exp(-(nes+stericEnergy+ionDCAEnergy+ionCavityEnergy + ionInterfaceEnergy)/kT_kJmol)
  #ionConcentration = concentration * exp(-getElectrostaticInteraction(ionCharge,u)/kT_kJmol)*exp(-nes/kT_kJmol)

  if(debug):
    
    stericV=project(stericEnergy,V)
    elsV=project(getElectrostaticInteraction(ionCharge,u),V)
    nesV=project(nes,V)
    ionV=project(ionConcentration,V)
    print("IonConcentration:\n")
    print("kT_kJmol="+str(kT_kJmol)+"  bulkConc="+str(concentration))
    print("ster",stericV.vector().array())
    print("els",elsV.vector().array())
    print("nes",nesV.vector().array())
    print("conc",ionV.vector().array())
  return ionConcentration

# returns total charge concentration profile in mol/L (moles of unit charge/L)
def TotalChargeDistribution(concList):
  charge = 0
  for i in range(0,len(concList)):
    #charge += opts.ionCharge[i] * opts.bulkConcentration[i] * opts.concentrationScale[i] * concList[i]
    # factor 2 corresponds to the ln2 used to normalise to 1
    charge += opts.ionCharge[i] * opts.bulkConcentration[i] * ( 2*exp(opts.concentrationScale[i]*concList[i]) - 1 )
  return charge

def ChargeDistributionFunction(concList):
  return ( TotalChargeDistribution(concList) * PoissonChargeFactor )

# collect form for entropic density (integrand to entropic energy)
# in concentration units (mol/L)
#
# scaledConcList is array of scaled concentrations (the conc component of the solution)
# bulkConcentrationList is usually the bulk concentrations (opts.bulkConcentration) but in principle a different set of bulk concentrations could be used instead (e.g. nonequilibrium conditions)
def entropicDensity (scaledConcList,bulkConcentrationList):
  entropicDensity = Constant (0.0)
  for i in range(len(scaledConcList)):
    cBulk = bulkConcentrationList[i]
    p = 2*exp(opts.concentrationScale[i]*scaledConcList[i]) - 1
    # rounding error may introduce probabilities like -1e-23
    # so force "negative" probabilities near 0 to 0 (leaving
    ionEntropicDensity = cBulk*(conditional(p<DOLFIN_EPS, 1, p*ln(p) - p + 1))
    entropicDensity += ionEntropicDensity
   
  return entropicDensity


class DisplacedBoundaryFunction(UserExpression):
  def  __init__(self, function, displacedPositionScaled, degree=2, element=None, **kwargs):
    self.my_function = function
    self.displacedPositionScaled = displacedPositionScaled
    self._degree=degree
    self._ufl_element = element
    super().__init__(**kwargs)

  def eval(self, values, xScaled):
    global separation
    print("displacedPositionScaled=",self.displacedPositionScaled )
    values[0] = self.my_function( xScaled + Point(self.displacedPositionScaled) )
  def value_shape(self):
    return ()

# this test form does work
def ChargeRegulatedSurfaceChargeSIt(concList):
  return concList[2]/(1.0+0.1*concList[2])*4e-2

def ChargeRegulatedSurfaceChargeSIs(concList):
  site = opts.bindingSites[0]
  return site.siteDensityPerMSq*qe*opts.ionCharge[2]*concList[2] / 10.0**(-site.ionCompetitors[0].pKdissociation) /(1.0+concList[2]/ 10.0**(-site.ionCompetitors[0].pKdissociation))

# throttle is applied to site densities
def ChargeRegulatedSurfaceChargeSI(siteList,scaledConcList,potential,throttle=1):
  #... updated concs not getting through when TrialFunction used
  # but "ufl/log.py, line 171, Linear forms must be defined using test functions only" when Function used :(
  # Function works with plain solve() but not with NonlinearVariationalProblem.solver.solve() ??
  # ... reproducible in minimal pb2 test: works via form with TrialFunction not Function, but surface charge from Function (i.e. concs updated)

  # ion concentrations are determined relative to current bulkConcentration (not surface reference)
  concList=[ opts.bulkConcentration[i] * ( 2*exp(opts.concentrationScale[i]*scaledConcList[i]) - 1 ) for i in range(len(scaledConcList)) ]
  #concList=scaledConcList
  
  totalSiteCharge=0.0

  for site in siteList:
    siteCharge=0.0
    associationFactor=1.0

    # surface "partial activity" means not the full electrochemical activity
    # it is defined as a = c_surf exp(+micro_NES/kT) = c_bulk exp(-qψ/kT)
    # c_surf and micro_NES require handling of the physisorption position
    # if DCA or Helmholtz layers are applied, which is difficult for direct computation using TrialFunctions (can't easily evaluated the values at the physisorption position)
    # So use the bulk/electrostatic formula instead.
    # Note c_bulk here is the reference concentration for the surface, which may
    # differ from the current actual bulk concentration under nonequilibrium conditions
    if site.bulkConcentrationSurfaceReference is None:
      bulkConcentrationSurfaceReference = opts.bulkConcentration
    else:
      bulkConcentrationSurfaceReference = site.bulkConcentrationSurfaceReference
    surfacePartialActivity = [ bulkConcentrationSurfaceReference[i]*exp(-getElectrostaticInteraction(opts.ionCharge[i],potential)/kT_kJmol) for i in range(len(bulkConcentrationSurfaceReference)) ]

    #doubleAssociation=numpy.zeros(len(site.ionCompetitors))
    doubleAssociation_site=[0] * len(site.ionCompetitors)  # use python list of zeros rather than numpy array to accomodate functions from boundDoubleConc
    doubleAssociation_charge=[0] * len(site.ionCompetitors)
    if (len(site.pKdoubleBoundDissociation)>0):
      for i in range(0,len(site.ionCompetitors)):
        ix=site.ionCompetitors[i].electrolyteIndex
        doubleAssociationValue_site = 0
        doubleAssociationValue_charge = 0
        for j in range(0,len(site.ionCompetitors)):
          jx=site.ionCompetitors[j].electrolyteIndex
          Ki = 10.0**(-site.ionCompetitors[i].pKdissociation)
          Kj = 10.0**(-site.ionCompetitors[j].pKdissociation)
          Kij = 10.0**(-site.pKdoubleBoundDissociation[i,j])
          Kji = 10.0**(-site.pKdoubleBoundDissociation[j,i])
          if pbconfig.useIncorrectDoubleBinding:
            inv_Kij_double = 1/(Ki*Kij)
          else:
            inv_Kij_double = 1/(Ki*Kij) + 1/(Kj*Kji)

          if numpy.isfinite(Kij):
            boundDouble_site = surfacePartialActivity[jx] / Kij
            doubleAssociationValue_site += boundDouble_site
          boundDouble_charge = surfacePartialActivity[ix]*surfacePartialActivity[jx] * inv_Kij_double
          doubleAssociationValue_charge += boundDouble_charge
        doubleAssociation_site[i] = doubleAssociationValue_site
        doubleAssociation_charge[i] = doubleAssociationValue_charge

    for i in range(0,len(site.ionCompetitors)):
      boundIon=0
      ix=(site.ionCompetitors[i]).electrolyteIndex
      if site.ionCompetitors[i].pKdissociation != -numpy.inf:
        Ki = 10.0**(-site.ionCompetitors[i].pKdissociation)
        boundIon = surfacePartialActivity[ix] / Ki
        if pbconfig.useIncorrectDoubleBinding and (len(site.pKdoubleBoundDissociation)>0):
          if numpy.isfinite(site.pKdoubleBoundDissociation[i,i]):
            doubleAssociation_charge[i] += surfacePartialActivity[ix]**2  / ( Ki * 10.0**(-site.pKdoubleBoundDissociation[i,i]) )
        associationFactor += boundIon * ( 1.0 + doubleAssociation_site[i])
      siteCharge += opts.ionCharge[ix]*(boundIon + doubleAssociation_charge[i])

    siteCharge /= associationFactor
    siteCharge += site.dissociatedCharge
    # the magnitude of (site density per MSq * qe) is around unity, so keep the two quantities together
    # otherwise if the 2 are treated separately (e.g. multiplying by qe at the end after summations) then the numerics renders the total charge near-zero, which is no good
    siteCharge *= (throttle*site.siteDensityPerMSq*qe)
    totalSiteCharge += siteCharge

  return totalSiteCharge


# convert the solution including charge profiles in format consistent
# with octave Poisson-Boltzmann implementation
#
# Converts the data from the saved XDMF file (HDF5 data content), since it is already ordered
def convertXDMF2OctavePB(separation, label=""):
  
  basename="finalPotential-L"+str(separation)+"nm"+label
  
  # pull octave data out of the HDF5 datasets for Xdmf rather than the Dolfin HDF5,
  # since they are already ordered.
  filenameXDMF_HD5 = basename + ".h5"
  if (MPI.rank(mpi_comm) == rootMPI):
    print( "reading XDMF-HDF5 from " + filenameXDMF_HD5)
  fileXDMF_HD5 = h5py.File(filenameXDMF_HD5, "r")
  
  # pull values out as numpy arrays
  angPerNM = 10.0
  
  # assume coordinates are identical for each dataset, take so them from the first set
  # note conversion from nm to Ang
  # XDMF 2.0:
  #coordsAng = fileXDMF_HD5.get("/Mesh/0/coordinates").value * angPerNM
  # XDMF 3.0 [dolfin 2016.2.0]: geometry includes 2nd dimension "y=0", so pick out the "x" value
  coordsAng = fileXDMF_HD5.get("/Mesh/0/mesh/geometry").value[:,[0]] * separation * angPerNM
  
  datasets = fileXDMF_HD5.get("/VisualisationVector")
  potential = datasets.get("0").value * opts.potentialScale
  scaledDfield = datasets.get("1").value * opts.potentialScale / opts.lengthScale
  totalCharge = datasets.get("2").value
  
  allData = numpy.concatenate( (coordsAng, potential, scaledDfield/angPerNM, totalCharge), axis=1 )
  
  commonDataCount = 3  # counting potential, scaledDfield, totalCharge
  ionCount = len(datasets.items()) - commonDataCount
  #ionCount -= 1  # adjust for chemisorbed charge
  #ionConc = []
  for i in range(ionCount):
    ionConc = opts.bulkConcentration[i] * ( 2*numpy.exp(opts.concentrationScale[i]*datasets.get(str(commonDataCount+i)).value) - 1 )
    allData = numpy.concatenate( (allData, ionConc), axis=1 )
      
  header="L(Ang)\tpotential(mV)\tscaledDfield\ttotalCharge(M)"
  for i in range(ionCount):
    header += "\tconc"+str(i)+"_"+opts.ionList[i].name+"[M]"

  #print "chemisorbed in ", str(commonDataCount+ionCount)
  #chemisorbedCharge = datasets.get(str(commonDataCount+ionCount)).value
  #allData = numpy.concatenate( (allData, chemisorbedCharge), axis=1 )
  #header += "\tchemisorbedCharge(m)"
  directory = opts.backgroundIonList[0].name + opts.backgroundIonList[1].name
  try:
        if not os.path.exists(directory):
            os.makedirs(directory)
  except OSError:
        print ('Directory already exists 1 ' +  directory)
  potentialFileName = '{}/leftPot{}-potDiff{}-potential.dat'.format(directory,opts.leftPotential, opts.potentialDifference)
  numpy.savetxt(potentialFileName, allData, '%.18g', '\t', '\n', header)
  # ~ returnArray = [leftElectrodeChargeDensity, RightElectrodeChargeDensity, totalChargeDensity, allData] 
  return allData #(leftElectrodeChargeDensity, RightElectrodeChargeDensity, totalChargeDensity, potential, coordsAng, allData) ###############################Added to try seek function
  


# print the solution including charge profiles to XDMF for plotting and HDF5 for reuse by dolfin
def printSolution( separation, u_all, nes, bulkConcentration, ionCharge, stericEnergyFloor, totalChargeDistributionFunction, V=[], label="" ):
  ulist = u_all.split()
  #print('U_ALL in PRINTSOLUTION is {}'.format(ulist))
  potential = ulist[0]
  #print('U_ALL in PRINTSOLUTION is {}'.format(potential))
  ionConcentrationProfiles = ulist[1:]

  Vpot=V.sub(0).collapse()
  if(Vpot and not ( Vpot==potential.function_space() or Vpot==potential.function_space() ) ):
    # some alien FunctionSpace provided, project onto it
    potential = project(potential,Vpot)
  else:
    Vpot=potential.function_space()
  #potential = project(opts.potentialScale*potential,Vpot)
  # collect auxiliary data (electric field, ion concentrations)
  # print scaledDfield  D' = (ε(r)/ε(0))E(r)  (with unit spatial dependence of ε(r) but without its magnitude)
  scaledDfield = project(-grad(potential),VectorFunctionSpace(Vpot.mesh(), "CG", 2))
  # ~ scaledDfield = inner(-grad(potential),FunctionSpace(Vpot.mesh(), "CG", 1))*dx
  # ~ scaledDfield = inner(-grad(potential),Vpot)*dx
  #print('U_ALL in PRINTSOLUTION is {}'.format(scaledDfield[0]))
  
  totalCharge = project(TotalChargeDistribution(ionConcentrationProfiles),FunctionSpace(Vpot.mesh(), "CG", 2))
 
  
  # Save solution to file
  basename="finalPotential-L"+str(separation)+"nm"+label
  
  # XDMF makes data more readily accessible for plotting, but is not fit for checkpointing
  # HDF5 stores the actual dolfin data so can be used for reprocessing or restarting
  filenameXDMF = basename+".xdmf"
  encoding = XDMFFile.Encoding.HDF5
  fileXDMF = XDMFFile(mpi_comm, filenameXDMF)
  
  filenameHDF5 = basename+"-dolfin"+".h5"
  fileHDF5 = HDF5File(mpi_comm, filenameHDF5, "w")
  
  # each function writes its own copy of the mesh,
  # but write a separate copy to make it clear
  fileXDMF.write(Vpot.mesh(),encoding) 
  fileHDF5.write(Vpot.mesh(),"mesh")

  # assign extra information to the HDF file
  # The dolfin API doesn't seem to let us manipulate HDF5 attributes for the root object
  # so stick them into the mesh group.
  # The dolfin API also doesn't let us add attributes to the XDMF file.
  attributesHDF5 = fileHDF5.attributes("mesh")
  attributesHDF5["separation"] = separation
  attributesHDF5["lengthUnits"] = "nm"
  attributesHDF5["potentialUnit"] = "mV"
  attributesHDF5["ionConcentrationUnits"] = "mol/L"

  # potential
  f=0
  fileXDMF.write(potential,f,encoding) 
  fileHDF5.write(potential,"potential")
  potentialAttributesHDF5 = fileHDF5.attributes("potential")
  potentialAttributesHDF5["potentialScale"] = opts.potentialScale
  
  # scaledDfield
  f+=1
  fileXDMF.write(scaledDfield,f,encoding) 
  fileHDF5.write(scaledDfield,"scaledDfield")

  # totalCharge
  f+=1
  fileXDMF.write(totalCharge,f,encoding) 
  fileHDF5.write(totalCharge,"totalCharge")

  # ion concentrations
  for i in range(0,len(ionCharge)):
    f+=1
    ionconc = ionConcentrationProfiles[i]
    #ionconc = project(opts.bulkConcentration[i]*ionConcentrationProfiles[i],Vpot)
    fileXDMF.write(ionconc,f,encoding) 
    fileHDF5.write(ionconc,"ionConcentration/"+str(i))
    concAttributesHDF5 = fileHDF5.attributes("ionConcentration/{}".format(i))
    concAttributesHDF5["charge"] = opts.ionCharge[i]
    concAttributesHDF5["bulkConcentration"] = opts.bulkConcentration[i]
    concAttributesHDF5["concentrationUnityFactor"] = opts.concentrationScale[i]
  fileHDF5.close()

  #chemisorbedCharge = project(SurfaceChemisorbedChargeConcentration( bindingSites, degree=2),V)
  #f+=1
  #fileXDMF.write(chemisorbedCharge,f,encoding) 

  # create datafile in octave pb format also
  del fileXDMF # there is no XDMFFile.close(), so need to del to read the file again
  # ~ convertXDMF2OctavePB( separation, label )

# transform function transforms x in [0,1]
# to a point in [a,b] biased on a power scale towards either a or b
# depending on whether x is above or below the cutpoint
#
# The cutpoint is also used to set the midpoint between a and b
def powerBiasTransform( x, a, b, scale ):
  return a + (b-a)*x**scale

def transformUnitToBiasedInterval(x, separation, twoBias=False):
  # set same number of output data points as input.
  xBiased = x*0
  
  # control parameters for the spread of biased points
  scale = 3


  if twoBias:
    midcut = 0.5
    midpoint = 0.5 * separation
  else:
    midcut = 1
    midpoint = separation
    
  mask = (x <= midcut)
  xBiased[ mask ] = powerBiasTransform( x[mask]/midcut, 0, midpoint, scale)
  mask = (x > midcut)
  xBiased[ mask ] = powerBiasTransform( (1-x[mask])/(1-midcut), separation, midpoint, scale)
  
  return xBiased

def getBaseEdgeMeshCoordinates(basePosition,N):
  unitRange = numpy.linspace(0,1,N)
  # spread over 0.1Å
  edgeSpread=0.01
  biasedCoordinates = transformUnitToBiasedInterval(unitRange, edgeSpread, False)
  # biasedCoordinates are pushed against 0, shift to either side of edge
  return numpy.append(biasedCoordinates+basePosition,basePosition-biasedCoordinates)
  
def getDCAEdgeMeshCoordinates(separation,N,twoBoundaries,useDCA):
  DCAedgeCoordinates = numpy.empty(0)
  for a in opts.gaussianRadius:
    ionEdgeCoords = getBaseEdgeMeshCoordinates(getHardSphereRadiusFromGaussian(a), N)
    DCAedgeCoordinates = numpy.append(DCAedgeCoordinates,ionEdgeCoords)
  if twoBoundaries and useDCA[1]:
    DCAedgeCoordinates = numpy.append(DCAedgeCoordinates,separation-DCAedgeCoordinates)
  return DCAedgeCoordinates

def getLayerEdgeMeshCoordinates(separation,N,twoBoundaries):
  # refactor to use pbconfig.boundaryConditionList.  How to set "direction" (add or subtract)?
  if not opts.outerHelmholtzPlane:
    edgeCoordinates = numpy.empty(0)
  else:
    edgeCoordinates = getBaseEdgeMeshCoordinates(opts.outerHelmholtzPlane, N)
    if twoBoundaries:
      edgeCoordinates = numpy.append(edgeCoordinates,separation-edgeCoordinates)
  return edgeCoordinates

def generateBiasedMeshCoordinates(separation, N, twoBoundaries):
  if (N%2==0):
    N += 1  # encourage an exact midpoint always
  unitRange = numpy.linspace(0,1,N)
  biasedCoordinates = transformUnitToBiasedInterval(unitRange, separation, twoBoundaries)
  fixedEdgeCoordinate = 0.5  # fixed fine mesh needed only out to 0.5 nm
  fixedEdge = transformUnitToBiasedInterval(unitRange, fixedEdgeCoordinate, False)
  if ( twoBoundaries ):
    fixedEdge = numpy.concatenate( (fixedEdge, separation - fixedEdge) )

  meshCoordinates =  numpy.append(biasedCoordinates,fixedEdge)

  # add a fine mesh at edges where appropriate
  # (ion DCAs, Helmholtz layers)
  edgeN=10
  if opts.useDCA:
    ionDCAedges = getDCAEdgeMeshCoordinates(separation,edgeN,twoBoundaries,opts.useDCA)
    meshCoordinates =  numpy.append(meshCoordinates,ionDCAedges)

  layerEdges = getLayerEdgeMeshCoordinates(separation,edgeN,twoBoundaries)
  if layerEdges.size:
    meshCoordinates =  numpy.append(meshCoordinates,layerEdges)

  # trim off excess points
  meshCoordinates=meshCoordinates[meshCoordinates>=0]
  meshCoordinates=meshCoordinates[meshCoordinates<=separation]

  # get unique coordinates in order
  coordinates = numpy.unique(numpy.sort(meshCoordinates))
  
  return coordinates

                                          
def generateBiasedMesh(separation, N, twoBoundaries):
  biasedCoordinates = generateBiasedMeshCoordinates(separation, N, twoBoundaries)
  mesh = UnitIntervalMesh(len(biasedCoordinates)-1)
  # work with the subset of mesh coordinates used in this MPI process
  # unit mesh contains "unit" coordinates in [0,1], convert to ordinal indices in [0,N]
  # and grab the corresponding entries in biasedCoordinates
  biasedMeshCoordinates = biasedCoordinates[ numpy.rint((mesh.coordinates() * (len(biasedCoordinates)-1))).astype(int) ]
  mesh.coordinates()[:,0] = biasedMeshCoordinates.flatten()/opts.lengthScale
  return mesh

def generateTwoBiasedMesh(separation, N):
  return generateBiasedMesh( separation, N, twoBoundaries=True )
    
def generateLeftBiasedMesh(separation, N):
  return generateBiasedMesh( separation, N, twoBoundaries=False )


def readOctaveMesh( filename ):
  data = numpy.loadtxt(filename)
  zz = data[:,0]
  mesh = UnitIntervalMesh(len(zz)-1)
  mesh.coordinates()[:,0] = zz/10.0
  return mesh
  
# evaluate Function f at Point point
# returns [ value, isValid ]
# where value = f(point)
# and isValid marks whether the value is valid or not
# (whether point is located in the local fragment of the mesh)
def evaluateAtPoint( f, point ):
  mesh = f.function_space().mesh()
  box = mesh.bounding_box_tree()
  #isValid = box.collides(point)
  #isValid = box.collides_entity(point) #disappeared in 2018.1.0 ??
  isValid = len(box.compute_entity_collisions(point)) != 0
  if (isValid):
    value = f(point)
  else:
    value = 0
  return [ value, isValid ]

# returns a single value f(point)
# the value is only valid for the root process (0)
# Returns 0 is the point is not contained in any part of the domain of f.
def evaluateAtTestedPoint( f, point ):
  dcomm = mpi_comm
  mcomm = MPI4.COMM_WORLD
  #pcomm = PETSc.Comm(mcomm)
  root = 0
  
  [ valueTentative, valueValid ] = evaluateAtPoint(f, point)
  valueTentativeA = numpy.array(range(1), numpy.double)
  valueTentativeA[0] = valueTentative
  values = numpy.array(range(MPI.size(dcomm)), numpy.double)
  mcomm.Gather( valueTentativeA, values )
  
  # hypatia MPI crashes if we try to gather bool values,
  # so convert validities to int instead
  valueValidA = numpy.array(range(1), numpy.int)
  valueValidA[0] = valueValid
  valueValidities = numpy.array(range(MPI.size(dcomm)), numpy.int)
  mcomm.Gather( valueValidA, valueValidities )

  testedValue=0
  if (MPI.rank(mpi_comm) == root):
    validityIndex = numpy.where(valueValidities==True)[0]
    if( validityIndex.size > 0 ):
      validIndex = validityIndex[0]
      testedValue = values[validIndex]
      isValid = valueValidities[validIndex]
    else:
      testedValue = 0
      isValid = False

  # get the root process to share the tested value with the other processes
  testedValueA = numpy.array(range(1), numpy.double)
  testedValueA[0] = testedValue
  mcomm.Bcast(testedValueA)
  testedValue = testedValueA[0]
              
  return testedValue

# copies the "y" values of the source into the target
# ("x" values may be different)
def copyFunctionValues(target, source):
  uflist = target.split(True)
  pflist = source.split(True)
  for i in range(len(uflist)):
    f = uflist[i]
    pf = pflist[i]

    # just reassign "y" values
    fv = f.vector()
    pfv = pf.vector()
    vec = Vector()

    newSize = f.function_space().dim()
    prevSize = pf.function_space().dim()
    pfv.gather(vec,numpy.array(range(prevSize),"intc"))
    #if numpy.isfinite(norm(vec)):
      # sometimes the data gathered is corrupt, in which don't set an initial guess
      # new and previous meshes may be different size. So interpolate.
    xi = numpy.array(range(newSize))/(newSize-1)
    xpi = numpy.array(range(prevSize))/(prevSize-1)
    newValuesAll = numpy.interp(xi,xpi,vec.array())
    # restrict data transfer to dofs on this process
    # cf. https://fenicsproject.org/qa/10005/setting-function-values-in-parallel-repeated-question?show=10005#q10005
    dofmap = f.function_space().dofmap()                                                             
    my_first, my_last = dofmap.ownership_range()
    x = f.function_space().tabulate_dof_coordinates()
    unowned = dofmap.local_to_global_unowned()
    dofs = list( filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned, range(my_last-my_first)) )
    newValues = newValuesAll[dofs]
    f.vector().set_local(newValues)
    f.vector().apply("insert")
    assign(target.sub(i), f)  # seems to work fine, this u_allf prints ok
    #else:
    #  print("[%d] corrupted initial guess, max val %g" % (i,max(abs(vec.array()))))
    #  hasInitialGuess=False
  return target

def runPB(separation, debyeLength, doPrintSolution=True):
  global previous_u_allf
  global concList
  global V1

  # bring all functions to "unity"
  
  global lastTime

  linearPotentialScale_mV = 1000*sc.Boltzmann * temperature / ( 2 * sc.e )
  opts.lengthScale = separation
  
  # fitting pKs
  if (MPI.rank(mpi_comm) == rootMPI):
    print( "conc=",opts.bulkConcentration)
  #parameters['form_compiler']['quadrature_degree'] =4
  #if ( abs(opts.leftPotential) > 500 ):
  #  baseMeshDensity = 10000
  
  # Create mesh and function space
  #mesh = IntervalMesh(baseMeshDensity,0,separation)
  #mesh = generateExponentialMesh(separation, debyeLength, baseMeshDensity)
  #mesh = generateTwoZoneBiasedMesh(separation, debyeLength, baseMeshDensity)
  mesh = generateTwoBiasedMesh(separation, baseMeshDensity)
  #mesh = generateLeftBiasedMesh(separation, baseMeshDensity)
  #mesh = readOctaveMesh("finalPotential-L100-octave.dat")
  #V = FunctionSpace(mesh, "CG", 1)
  
  P1 = FiniteElement('CG', interval, 2)
  elist = [ P1 ]  # for potential
  for i in range(len(opts.bulkConcentration)):
    elist.append( P1 )
  element = MixedElement(elist)
  V = FunctionSpace(mesh, element)
  V1 = FunctionSpace(mesh, P1)

  opts.boundary=[]
  opts.boundary.append(LeftBoundary())
  opts.boundary.append(RightBoundary(separation))
  
  #boundaries = FacetFunction("size_t", mesh)
  boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
  boundaries.set_all(0)
  if not opts.symmetricBoundary:
    opts.boundary[1].mark(boundaries, 1)
  ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
  opts.surfaceMeasure_ds=ds

  zeroFunction=Constant(0)
  noNES = [zeroFunction for i in range(len(opts.bulkConcentration))]
  nesEl=P1

  nesEnergy = list( TwoSurfaceDispersionEnergy(ion.dispersionB, ion.vacuumDispersionB, ion.gaussianRadiusAng/10,ion.charge,element=nesEl) for ion in opts.ionList )
  
  nes=noNES  # first get electrostatic solution to initialise dispersion solution
  #nes=nesEnergy
  opts.ionNESinteraction=nes
  foundElectrostaticSolution=False

  #  ....ought to add steric energy
  #concentration=bulkConcentration
  #concentration=cellConcentration()

  opts.concentrationScale = numpy.log(2)*numpy.ones(len(opts.bulkConcentration))
  opts.potentialScale = getReferencePotential( pbconfig.boundaryConditionList[0] )
  if (MPI.rank(mpi_comm) == rootMPI):
    print("main run opts.concentrationScale=",opts.concentrationScale)
    print("potential scale = {}".format( opts.potentialScale))
    print("potential difference = {}".format(opts.potentialDifference))
  #exit(0)
  
  ionicStrength = getIonicStrength(opts.bulkConcentration,opts.ionCharge)


#========
#... this scaling using (prob-1)/prob(0) to represent "concentration" as unit (1 at boundary, 0 in bulk)
#seems to help a little,  that is abs residual in 2nd iteration falls to 1 (-500 mV)
#but rel residual is lower smaller by 1e10 and that's what needs balancing
#Is it the length scale?


  
  # Define variational problem
  u_allt = TrialFunction(V)
  v = TestFunction(V)

  u_allf=Function(V)
  u_allf_blank=u_allf.copy(True)
  hasInitialGuess=False
  # starting from previous solution is giving indeterminate results, so deactivate
  #if False:
  #if previous_u_allf:
  #  # seed solution using previous
  #  u_allf = copyFunctionValues(u_allf, previous_u_allf)
  #  hasInitialGuess=True
  #else:
  #  initialDH = project(DebyeHuckelTwoSide(debyeLength,opts.leftPotential,element=P1),V1)
  #  #u_allf = copyFunctionValues(u_allf, previous_u_allf)
  #  #uflist = split(u_allf)
  #  assign(u_allf.sub(0),initialDH)
  #  hasInitialGuess=True
   
  utlist = split(u_allt)  # runs but solution doesn't seem to match test without MixedSpace
  #ulist = split(u_all)  # fails: ufl/log.py, line 171, Linear forms must be defined using test functions only: (1, 'k', 15, 15)
  #ulist = u_all.split() # gives Error:   Unable to successfully call PETSc function 'MatSetValuesLocal'.
  pot = utlist[0]
  uflist = split(u_allf)  # fails: ufl/log.py, line 171, Linear forms must be defined using test functions only: (1, 'k', 15, 15)
  concList = utlist[1:]  # pot from u and concList from u_all?  Doesn't seem to matter if concList is from either u or u_all.

  #... how to get Efield and ChargeRegulatedSurfaceChargeSI to work with MixedSpace ??
  #concListVal=[]
  #for i in range(len(concList)):
  #  concListVal.append( project( concList[i], V.sub(i+1) ) )

  isConverged=False
  while (not foundElectrostaticSolution) or (not isConverged):
    for i in range(len(opts.bulkConcentration)):
      #print('LEFTPOTENTIAL in runPB is {}'.format(opts.leftPotential))
      surfaceProbabilityFunc = project(IonConcentration(1,opts.leftPotential,nes[i],opts.ionCharge[i],opts.stericEnergyFloor[i],opts.ionDCAenergy[i],opts.ionCavityEnergy[i],opts.ionInterfaceEnergy[i], V),V1)
      surfaceProbability=evaluateAtTestedPoint(surfaceProbabilityFunc,Point(0))
      # apply ln(2) to get coion background value at "1"
      concScale = max( ln(2), ln(surfaceProbability+1)-ln(2) )
      opts.concentrationScale[i] = concScale

    x = SpatialCoordinate(mesh)
    if opts.innerRadius < 0:
      # planar system
      r=Constant(1)
    else:
      # spherical system
      r = Expression("a+L*x[0]",a=opts.innerRadius,L=opts.lengthScale,degree=1)
    # Boltzmann equations
    ab = 0
    for i in range(len(opts.bulkConcentration)):
      #ab +=  inner(concList[i], v[i+1])*dx - inner(IonConcentration(opts.bulkConcentration[i],pot,nes[i],opts.ionCharge[i],opts.stericEnergyFloor[i], V.sub(0)), v[i+1])*dx
      #ab +=  inner(concList[i], v[i+1])*dx - inner(IonConcentration(1/opts.concentrationScale[i],opts.potentialScale*pot,nes[i],opts.ionCharge[i],opts.stericEnergyFloor[i], V.sub(0)), v[i+1])*dx
      ab +=  r*r*opts.lengthScale*inner(concList[i], v[i+1])*dx - r*r*opts.lengthScale*inner((ln(IonConcentration(1,opts.potentialScale*pot,nes[i],opts.ionCharge[i],opts.stericEnergyFloor[i], opts.ionDCAenergy[i], opts.ionCavityEnergy[i],opts.ionInterfaceEnergy[i], V.sub(0))+1)-ln(2))/opts.concentrationScale[i], v[i+1])*dx

    # Debye-Hückel
    #f=-pot/debyeLength**2
    # Poisson equation
    #f=ChargeDistributionFunction(concList)
    f=ChargeDistributionFunction(concList)*Constant(1/opts.potentialScale)

    ap = r*r*inner(nabla_grad(pot), nabla_grad(v[0]))/opts.lengthScale*dx
    Lp = r*r*f*opts.lengthScale*v[0]*dx

    # Define boundary condition
    targetBoundaryValue=pbconfig.boundaryConditionList[0].value #  ... rework from here for general case (dirichlet or neumann)
    targetPotential = opts.potentialScale
    # Compute solution
    isConverged = False
    #reductionFactor=5
    #amplificationFactor=1.1  # be conservative once a safe solution is obtained, let amplificationFactor be less than reductionFactor
    boundaryFactor = 0.25  # factor 100 ramping seems fine for this alumina. Need way low (<1e-3) to catch the first dispersion solution, but once you have it you can ramp up straight to the target
    downFactor=3 # accelerate boundaryFactor when reducing boundaryThrottle
    #boundaryValue=targetBoundaryValue*reductionFactor  # multiply by reductionFactor so first attempt uses targetBoundaryValue
    # 42 is the point at which the universe starts to push back
    # don't try to start higher than it
    throttleTrigger = 42*linearPotentialScale_mV
    boundaryThrottle=min(1,(throttleTrigger/abs(targetPotential)))/(0.999*boundaryFactor*downFactor)  # multiply by factor so first attempt uses targetBoundaryValue
    if not foundElectrostaticSolution:
      #boundaryThrottle *= 0.9   # regulated charge is unstable when ion dispersion is active
      #boundaryThrottle *= 0.3   # regulated charge is unstable when ion dispersion is active
      boundaryThrottle *= 1   # regulated charge is stable
    else:
      boundaryThrottle = 0.6*(1-boundaryFactor)/(1-boundaryFactor)**2   # electrostatic solution found, jump straight into final dispersion solution. Use "x/x^2" to get clean boundaryThrottle below as 1.0 rather than 0.9999999999999
      #boundaryThrottle = 1*(1-boundaryFactor)/(1-boundaryFactor)**2 

    hasTestedPreviousMesh=False
    if previous_u_allf:
      # get occasional bifurcations from one separation to the next, (which is a bit of a worry. Is it because the mesh is not yet adaptive?)
      # especially with large potentials, unless we propagate over the solution for the previous separation
      previousConvergedSolution = Function(V)
      # but standard project or interpolate does not yet work in parallel
      # so have to use MiroK's fenictools, or use LagrangeInterpolator
      LagrangeInterpolator.interpolate(previousConvergedSolution,previous_u_allf)
    else:
      previousConvergedSolution = False
      hasTestedPreviousMesh=True  # "has tested" in the sense that there is no previous mesh left untested
    #while (not isConverged) or abs(boundaryValue) < abs(targetBoundaryValue):
    fullThrottleAttempts = 0
    while (not isConverged) or boundaryThrottle < 1:
    #while (not isConverged) or abs(boundaryValue) < abs(75):

      if isConverged:
        #boundaryValue *= amplificationFactor
        #boundaryValue *= (1+boundaryFactor)
        #boundaryValue = numpy.sign(targetBoundaryValue)*min(abs(boundaryValue),abs(targetBoundaryValue))
        boundaryThrottle *= (1+boundaryFactor)
        boundaryThrottle = min(1,boundaryThrottle)
      else:
        #boundaryValue /= reductionFactor
        #boundaryValue *= (1-boundaryFactor)
        boundaryThrottle *= boundaryFactor*downFactor
        # "if opts.potentialScale > 1" ...  use a clean slate for near-zero systems
        # to minimise "oscillations" near zero
        if opts.potentialScale > 1 and previousConvergedSolution:
          u_allf = previousConvergedSolution.copy(deepcopy=True)
        else:
          u_allf = Function(V) # start afresh until converged
      boundaryThrottle = min(1,boundaryThrottle)  # just to be safe, guard against overshooting
      if (MPI.rank(mpi_comm) == rootMPI):
        print( "boundary throttle = {}".format(boundaryThrottle))

      if boundaryThrottle >= 1:
        fullThrottleAttempts += 1
      if fullThrottleAttempts > 2:
        if (MPI.rank(mpi_comm) == rootMPI):
          print("boundary throttle is going round in circles, give up now")
        return (u_allf, False)
      if (boundaryThrottle < 1e-5):
        if (MPI.rank(mpi_comm) == rootMPI):
          print("boundary throttle has incontinence, give up now")
        return (u_allf, False)

      constantPotentialBC = []
      chargeBC = 0
      for b in pbconfig.boundaryConditionList:
        surfaceCharge = 0
        if b.type == BoundaryConditionType.CONSTANT_POTENTIAL:
          # Dirichlet constant potential given in mV
          abc = DirichletBC(V.sub(0), boundaryThrottle*b.value/opts.potentialScale, opts.boundary[b.domainIndex])
          constantPotentialBC.append(abc)
        elif b.type == BoundaryConditionType.CONSTANT_CHARGE:
          # Neumann constant charge given in C/m^2
          surfaceCharge = boundaryThrottle*b.value / EPS_0 / EPS_VAC * 1000 * 1e-9 / opts.potentialScale
        elif b.type == BoundaryConditionType.CHARGE_REGULATED:
          # Neumann(Robin) regulated charge in C/m^2
          # given as list of regulated sites (must using TrialFunctions here, not Functions, i.e. from utlist not uflist)
          surfaceCharge = ChargeRegulatedSurfaceChargeSI(b.value,concList,opts.potentialScale*pot,boundaryThrottle) / EPS_0 / EPS_VAC * 1000 * 1e-9 / opts.potentialScale
          #surfaceCharge = ChargeRegulatedSurfaceChargeSI(b.value,utlist[1:],opts.potentialScale*utlist[0],boundaryThrottle) / EPS_0 / EPS_VAC * 1000 * 1e-9 / opts.potentialScale
          #surfaceCharge = ChargeRegulatedSurfaceChargeSI(b.value,uflist[1:],opts.potentialScale*uflist[0],boundaryThrottle) / EPS_0 / EPS_VAC * 1000 * 1e-9 / opts.potentialScale
        if surfaceCharge!=0:
          chargeBC += r*r*surfaceCharge*v[0]*ds(b.domainIndex)
######################################################################################################################################################################################################################
      F=ab+ap-Lp-chargeBC

      F=action(F,u_allf)
      J=derivative(F,u_allf,u_allt)    

      # Define boundary condition

      mainMeshAdaptive=False
      adaptiveTol=1.0e-4

      problem = NonlinearVariationalProblem(F, u_allf, constantPotentialBC, J)
      if (mainMeshAdaptive):
        # Define goal functional (quantity of interest)
        M = u_allf*dx()
        solver  = AdaptiveNonlinearVariationalSolver(problem,M)
        solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
      else:
        solver  = NonlinearVariationalSolver(problem)
      prm = solver.parameters
      # for -600 mV, only SNES works so far
      #prm['nonlinear_solver']='snes'
      #if hasInitialGuess:
      #  # 1e-8 fails e.g. at L=108nm
      #  tolerance = 1e-7
      #else:
      # in same cases 1e-7 fails e.g. L=131nm
      #tolerance = 1e-6
      if opts.potentialScale < 0.1:
        absTol = 1e-10
      elif opts.potentialScale < 100:
        absTol = 1e-5
      else:
        absTol = 1e-4
      #relTol = 1e-2
      relTol = 1e-30  # I don't trust relative tolerance. Switch it "off".
      if separation < 1.5:  # get over the divergence hump at L=1nm
        absTol *= 5

      prm['newton_solver']['error_on_nonconvergence'] = False
      prm['newton_solver']['absolute_tolerance'] = absTol
      prm['newton_solver']['relative_tolerance'] = relTol
      prm['newton_solver']['maximum_iterations'] = 10
      #prm['newton_solver']['maximum_iterations'] = 50
      prm['snes_solver']['absolute_tolerance'] = absTol
      prm['snes_solver']['relative_tolerance'] = relTol
      #prm['snes_solver']['maximum_iterations'] = 50
      #prm['newton_solver']['linear_solver'] = 'petsc'


      iterative_solver=False
      #iterative_solver=True
      if iterative_solver:
        #prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
        prm['newton_solver']['linear_solver'] = 'gmres'
        #prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['preconditioner'] = 'hypre_parasails'
        krylovSolver=prm['newton_solver']['krylov_solver']
        krylovSolver['absolute_tolerance'] = absTol
        krylovSolver['relative_tolerance'] = relTol
        #krylovSolver['maximum_iterations'] = 20
        #krylovSolver['gmres']['restart'] = 40
        #krylovSolver['preconditioner']['ilu']['fill_level'] = 0

      if previousConvergedSolution:
        prm['newton_solver']['maximum_iterations'] = 20
      if isConverged:
        prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
      try:
        (niter,isConverged) = solver.solve()
      except:
        isConverged=False
      if isConverged:
        previousConvergedSolution = u_allf.copy(deepcopy=True)
      else:
        if not hasTestedPreviousMesh:
          # has now tried previous mesh and failed. Restart from scratch.
          hasTestedPreviousMesh = True
          previousConvergedSolution = False

    if (not foundElectrostaticSolution) and isConverged:
        foundElectrostaticSolution = True
        if not opts.electrostaticOnly:
          if (MPI.rank(mpi_comm) == rootMPI):
            print("Found electrostatic solution, adding dispersion now")
          isConverged = False
          nes=nesEnergy
          opts.ionNESinteraction=nes
          
  if doPrintSolution and isConverged and (opts.printAllSolutions or numpy.isclose(separation,debyeLength)):
	  try:
		  site=pbconfig.boundaryConditionList[0].value[0]
		  siteValue=site.siteDensityPerNMSq
	  except:
		  siteValue = pbconfig.boundaryConditionList[0].value
		  #printSolution( separation, u_allf.leaf_node(), nes, opts.bulkConcentration, opts.ionCharge, opts.stericEnergyFloor, TotalChargeDistribution, V, "leftPotential{}-salt{}M".format(opts.leftPotential,opts.backgroundSaltConc ) )
	  printSolution( separation, u_allf.leaf_node(), nes, opts.bulkConcentration, opts.ionCharge, opts.stericEnergyFloor, TotalChargeDistribution, V, "-leftPotential{}-salt{}M".format(opts.leftPotential,opts.backgroundSaltConc ) )
  return (u_allf,isConverged)





def getPotentialForSiteDensityPerNMSq( siteDensityPerNMSq ):
  siteAlumina=pbconfig.boundaryConditionList[0].value[0]
  siteAlumina.siteDensityPerNMSq = siteDensityPerNMSq
  siteAlumina.siteDensityPerMSq = siteAlumina.siteDensityPerNMSq * 1e18
  (u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=False)
  pot = opts.potentialScale*evaluateAtTestedPoint(u_allf.split()[0],Point(0))
  if (MPI.rank(mpi_comm) == rootMPI):
    print("{} {} {} {}".format(siteDensityPerNMSq, pot, opts.targetPotential, opts.targetPotential-pot))
  return pot



def seekChargeRegulatedPotential( separation, debyeLength):
  finalSiteDensityPerNMSq, results = scipy.optimize.brentq(lambda s: getPotentialForSiteDensityPerNMSq(s)-opts.targetPotential, 0, 3, full_output=True)
  if (MPI.rank(mpi_comm) == rootMPI):
    print("found best site density per NMSQ = {}".format(finalSiteDensityPerNMSq))
    print(results)
  (u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=True)
  return (u_allf,isConverged)

####################################################################################################################################################

def getTotalElectronChargeDensity(leftPotential):#, potentialDifference):
	global separation
	global allPrintedData             # added because of the error: local variable 'allPrintedData' referenced before assignment
	getBoundaryCondition(leftPotential,opts.potentialDifference,opts.symmetricBoundary)
	opts.leftPotential = leftPotential
	
	debyeLength = getDebyeLength(opts.bulkConcentration,opts.ionCharge)
	
	(u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=False)
	ulist = u_allf.split()
	potential = ulist[0]
	
	Vpot=potential.function_space().collapse()
	angPerNM = 10.0
	# ~ scaledDfield1 = project(-grad(potential),VectorFunctionSpace(Vpot.mesh(), "CG", 2))# * opts.potentialScale / opts.lengthScale /angPerNM
	# ~ scaledDfield2 = project(-Dx(potential,0),Vpot)# * opts.potentialScale / opts.lengthScale /angPerNM
	scaledDfield = project(-potential.dx(0),Vpot)# * opts.potentialScale / opts.lengthScale /angPerNM
	
	surfaceCharge = ChargeRegulatedSurfaceChargeSI(b.value,concList,opts.potentialScale*pot,boundaryThrottle) / EPS_0 / EPS_VAC * 1000 * 1e-9 / opts.potentialScale
	
	leftElectrodeChargeDensity = sc.epsilon_0 * EPS_0 * 10**7 * evaluateAtTestedPoint( scaledDfield, Point(0) ) * opts.potentialScale / opts.lengthScale /angPerNM
	leftElectronChargeDensity = leftElectrodeChargeDensity - leftBoundCharge
	RightElectrodeChargeDensity = -sc.epsilon_0 * EPS_0 * 10**7 * evaluateAtTestedPoint( scaledDfield,Point(separation / opts.lengthScale) ) * opts.potentialScale / opts.lengthScale /angPerNM
	RightElectronChargeDensity = RightElectrodeChargeDensity - RightBoundCharge
	totalElectronChargeDensity = leftElectronChargeDensity + RightElectronChargeDensity 
	return totalElectronChargeDensity


def getTotalChargeFromPD(potentialDifference):#, potentialDifference):
	global separation
	global allPrintedData         # added because of the error: local variable 'allPrintedData' referenced before assignment
	getBoundaryCondition(opts.leftPotential,potentialDifference,opts.symmetricBoundary)
	opts.potentialDifference = potentialDifference
	baseMeshDensity=1000
	
	debyeLength = getDebyeLength(opts.bulkConcentration,opts.ionCharge)
	
	(u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=False)
	ulist = u_allf.split()
	potential = ulist[0]
	
	Vpot=potential.function_space().collapse()
	angPerNM = 10.0	
	# ~ scaledDfield = project(-grad(potential),VectorFunctionSpace(Vpot.mesh(), "CG", 2))
	# ~ scaledDfield = project(-Dx(potential,0),Vpot)
	scaledDfield = project(-potential.dx(0),Vpot)
	
	leftElectrodeChargeDensity = sc.epsilon_0 * EPS_0 * 10**7 * evaluateAtTestedPoint( scaledDfield, Point(0) ) * opts.potentialScale / opts.lengthScale /angPerNM
	RightElectrodeChargeDensity = -sc.epsilon_0 * EPS_0 * 10**7 * evaluateAtTestedPoint( scaledDfield,Point(separation / opts.lengthScale) ) * opts.potentialScale / opts.lengthScale /angPerNM
	totalElectrodeChargeDensity = leftElectrodeChargeDensity + RightElectrodeChargeDensity
	return totalElectrodeChargeDensity
	

def seekChargeBalancedPotential(potentialDifference):
	if (potentialDifference == 0):
		lowerBound = -20
		upperBound = 20
	elif (potentialDifference > 1000 and potentialDifference < 1500):
		lowerBound = 0.75 * potentialDifference             
		upperBound = 0.85 * potentialDifference
		
	elif (potentialDifference > 1500):
		lowerBound = potentialDifference
		upperBound = 0.85 * potentialDifference
	
	else:
		lowerBound = potentialDifference/2
		upperBound = 0.65 * potentialDifference
	
	while True:
		lowerBound = 0.85 * lowerBound
		if(getTotalElectronChargeDensity(lowerBound) < 0):
			break
	while True:
		upperBound = 1.05 * upperBound
		if(getTotalElectronChargeDensity(upperBound) > 0):
			break
			
		
	leftPotential, totalChargeDensity = scipy.optimize.brentq(lambda s: getTotalElectronChargeDensity(s), lowerBound, upperBound, xtol=1e-3, full_output=True)
	# ~ if (MPI.rank(mpi_comm) == rootMPI):
		# ~ print("leftPotential = {}".format(leftPotential))
		# ~ print("totalChargeDensity from SEEK = {}".format(totalChargeDensity))
	
	return leftPotential


def seekChargeBalancedPotentialFromPD( lowerBound, upperBound):
	potentialDifference, totalChargeDensity = scipy.optimize.brentq(lambda s: getTotalChargeFromPD(s), lowerBound, upperBound, xtol=1e-3, full_output=True)
		
	return potentialDifference

# ~ def seekChargeBalancedPotentialFromPD( potentialDifference):
	# ~ lowerBound = 0.95 * potentialDifference
	# ~ upperBound = 1.05 * potentialDifference
	
	# ~ while True:
		# ~ lowerBound = 0.95 * lowerBound
		# ~ if(getTotalChargeFromPD(lowerBound) < 0):
			# ~ break
	# ~ while True:
		# ~ upperBound = 1.05 * upperBound
		# ~ if(getTotalChargeFromPD(upperBound) > 0):
			# ~ break
			
	# ~ potentialDifference, totalChargeDensity = scipy.optimize.brentq(lambda s: getTotalChargeFromPD(s), lowerBound, upperBound, xtol=1e-3, full_output=True)
	# ~ if (MPI.rank(mpi_comm) == rootMPI):
		# ~ print("leftPotential = {}".format(leftPotential))
		# ~ print("totalChargeDensity from SEEK = {}".format(totalChargeDensity))
	
	# ~ return potentialDifference

####################################################################################################################################################
#def getPotentialForpH( pH, siteDensity_NM2, pK_HH, printSolution=False ):
#def getPotentialForpH( pH, pK_HH, printSolution=False ):
#def getPotentialForpH( pH, siteDensity_NM2, pK_HH, printSolution=False ):
#def getPotentialForpH( pH, pK_H, printSolution=False ):
def getPotentialForpH( pH,  siteDensity_NM2, pK_HH, pK_H, printSolution=False ):
  global separation
  pot = numpy.zeros(len(pH))
  for i in range(len(pH)):
    opts.pH=pH[i]
    siteAlumina=pbconfig.boundaryConditionList[0].value[0]
    siteAlumina.siteDensityPerNMSq = siteDensity_NM2
    siteAlumina.siteDensityPerMSq = siteAlumina.siteDensityPerNMSq * 1e18
    siteAlumina.pKdoubleBoundDissociation = numpy.array([[pK_HH]])
    siteAlumina.ionCompetitors[0].pKdissociation = pK_H

    updateElectrolyte()
    debyeLength=getDebyeLength(opts.bulkConcentration,opts.ionCharge)
    separation=20*debyeLength
    (u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=printSolution)
    pot[i] = opts.potentialScale*evaluateAtTestedPoint(u_allf.split()[0],Point(0))
  if (MPI.rank(mpi_comm) == rootMPI):
    print("({}, {}, {}, {}, {})".format(pH,pot,siteAlumina.siteDensityPerNMSq,siteAlumina.pKdoubleBoundDissociation[0,0], siteAlumina.ionCompetitors[0].pKdissociation))
  return pot
  


def fitSurfaceParametersToZeta():
  # fit to array of zeta potential data
  # the complicated form here enables targetIndices to be either
  # an array(=selected values) or None(=all values)
  targetpH = opts.targetPotential[:,0][opts.targetIndices].flatten()
  targetPotential = opts.targetPotential[:,1][opts.targetIndices].flatten()
  #pInitialGuess = [ 0.18690302344972295 ]  # sites per nm^2
  #pInitialGuess = [ 8 ]  #  pK_HH
  #pInitialGuess = [ 0.20198272457895422, 8.120762408324598 ]  # sites per nm^2, pK_HH
  #pInitialGuess = [ 0.2103858777865185, 6.143211246065577 ]  # sites per nm^2, pK_HH
  #pInitialGuess = [ 10.790785181379848 ]  #  pK_H
  #pInitialGuess = [ 0.20313810645510466, 6.17788647918279, 10.211008497679284 ]  # sites per n30*m^2, pK_HH, pK_H
  pInitialGuess = [ 0.15808349327048268, 8.370779829900533, 10.249947473146204 ]  # sites per nm^2, pK_HH, pK_H
  popt = scipy.optimize.curve_fit( lambda pH, Ns, pKHH, pKH : getPotentialForpH( pH, Ns, pKHH, pKH ), targetpH, targetPotential, p0=pInitialGuess)
  if (MPI.rank(mpi_comm) == rootMPI):
    print("Seeking to match experimental data:",  opts.targetPotential[opts.targetIndices,:])
    print("found best parameters = ",popt)
  return popt
    
# returns site entropic and other nonelectrostatic contributions to the chemisorption free energy of a charge regulated site
# energy returned in microJ/m^2
def getFreeEnergyChemicalNonelectrostaticForSingleChargeRegulatedSurface(siteList,scaledConcList,potential,nes):
  energyChemicalNonelectrostaticDontUse=0;
  energySiteEntropic=0;
  energyChemicalCompetitive=0;

  energySiteEntropicCorrection=0;
  energyTwoBind=0;

  boundChargeSI=0.0

  concList=[opts.bulkConcentration[i] * ( 2*exp(opts.concentrationScale[i]*scaledConcList[i]) - 1 ) for i in range(len(scaledConcList)) ]
  
  for site in siteList:
    siteCharge=0.0
    associationFactor=1.0
    
    siteEnergyChemicalNonelectrostaticDontUse = 0;  # likely need to remove this once the chemisorption NES corrections are complete

    
    # surface "partial activity" means not the full electrochemical activity
    # it is defined as a = c_surf exp(+micro_NES/kT) = c_bulk exp(-qψ/kT)
    # c_surf and micro_NES require handling of the physisorption position
    # if DCA or Helmholtz layers are applied, which is difficult for direct computation using TrialFunctions (can't easily evaluated the values at the physisorption position)
    # So use the bulk/electrostatic formula instead.
    # Note c_bulk here is the reference concentration for the surface, which may
    # differ from the current actual bulk concentration under nonequilibrium conditions
    if site.bulkConcentrationSurfaceReference is None:
      bulkConcentrationSurfaceReference = opts.bulkConcentration
    else:
      bulkConcentrationSurfaceReference = site.bulkConcentrationSurfaceReference
    surfacePartialActivity = [ bulkConcentrationSurfaceReference[i]*exp(-getElectrostaticInteraction(opts.ionCharge[i],potential)/kT_kJmol) for i in range(len(bulkConcentrationSurfaceReference)) ]
    
    amountAdsorbedUnnormalised = [0] * len(site.ionCompetitors)
    doubleAssociation_site=[0] * len(site.ionCompetitors)  # use python list of zeros rather than numpy array to accommodate functions from boundDoubleConc
    doubleAssociation_charge=[0] * len(site.ionCompetitors)
    Kdouble=numpy.empty(0)
    inv_K_double=numpy.empty(0)
    if (len(site.pKdoubleBoundDissociation)>0):
      Kdouble = 10**( -numpy.array(site.pKdoubleBoundDissociation) )
      inv_K_double = numpy.zeros(Kdouble.shape)
      for i in range(0,len(site.ionCompetitors)):
        ix=site.ionCompetitors[i].electrolyteIndex
        doubleAssociationValue_site = 0
        doubleAssociationValue_charge = 0
        for j in range(0,len(site.ionCompetitors)):
          jx=site.ionCompetitors[j].electrolyteIndex
          Ki = 10.0**(-site.ionCompetitors[i].pKdissociation)
          Kj = 10.0**(-site.ionCompetitors[j].pKdissociation)
          Kij = Kdouble[i,j]
          Kji = Kdouble[j,i]
          if numpy.isfinite(Kij):
            boundDouble_site = surfacePartialActivity[jx] / Kij
            doubleAssociationValue_site += boundDouble_site
          if pbconfig.useIncorrectDoubleBinding:
            inv_Kij_double = 1/(Ki*Kij)
          else:
            inv_Kij_double = 1/(Ki*Kij) + 1/(Kj*Kji)
          inv_K_double[i][j] = inv_Kij_double
          boundDouble_charge = surfacePartialActivity[ix]*surfacePartialActivity[jx] * inv_Kij_double
          doubleAssociationValue_charge += boundDouble_charge
          
        doubleAssociation_site[i] =  doubleAssociationValue_site
        doubleAssociation_charge[i] =  doubleAssociationValue_charge

    for i in range(0,len(site.ionCompetitors)):
      if site.ionCompetitors[i].pKdissociation != -numpy.inf:
        ix=(site.ionCompetitors[i]).electrolyteIndex
        Ki = 10.0**(-site.ionCompetitors[i].pKdissociation)
        boundIon = surfacePartialActivity[ix] / Ki
        if pbconfig.useIncorrectDoubleBinding and (len(site.pKdoubleBoundDissociation)>0):
          if site.pKdoubleBoundDissociation[i,i] != -numpy.inf:
            doubleAssociation_charge[i] += surfacePartialActivity[ix]**2 / (Ki * 10.0**(-site.pKdoubleBoundDissociation[i,i]) )
        amountAdsorbedUnnormalised[i] =  boundIon  + doubleAssociation_charge[i]
        siteCharge += opts.ionCharge[ix]*amountAdsorbedUnnormalised[i]
        associationFactor += boundIon * ( 1.0 + doubleAssociation_site[i])

        siteEnergyChemicalNonelectrostaticDontUse += amountAdsorbedUnnormalised[i] * nes[i]
    
    siteEnergyChemicalNonelectrostaticDontUse /= associationFactor
    
    siteCharge /= associationFactor
    siteCharge += site.dissociatedCharge
    # the magnitude of (site density per MSq * qe) is around unity, so keep the two quantities together
    # otherwise if the 2 are treated separately (e.g. multiplying by qe at the end after summations) then the numerics renders the total charge near-zero, which is no good
    siteCharge *= (site.siteDensityPerMSq*qe)
    boundChargeSI += siteCharge

    ## get reference conditions
    nativeIonCompetitorIndex=site.nativeIonCompetitorIndex;  # defaults to 0, first ion in competitor list
    nativeIonElectrolyteIndex = site.ionCompetitors[ nativeIonCompetitorIndex ].electrolyteIndex
    Kv = 10**( -site.ionCompetitors[ nativeIonCompetitorIndex ].pKdissociation );
    
    if ( nativeIonElectrolyteIndex > 0 ):
      nativeIonCharge = opts.ionCharge[ nativeIonElectrolyteIndex ]
    else:
      nativeIonCharge = site.implicitNativeIonCharge;   # (defaults to implicit H+)

    if Kdouble.size > 0:
      Kvv = Kdouble[nativeIonCompetitorIndex,nativeIonCompetitorIndex];
    else:
      Kvv = numpy.inf;   # no double binding

    # amount of reference ion bound (at reference), in units of site density Ns
    if ( ( not site.dissociatedCharge==0) and (not nativeIonCharge == -site.dissociatedCharge) and (len(Kdouble) > 0) ):
      # complex case of amphoteric site with asymmetric  charge : simple reference binding not available
      # two possible solutions in ±√
      refLead = Kv*(1+site.dissociatedCharge/nativeIonCharge);
      refDenom = 2*Kv*Kvv*(2+site.dissociatedCharge/nativeIonCharge);
      refRoot = sqrt( refLead**2  - 2*refDenom*site.dissociatedCharge/nativeIonCharge );
      referenceActivity1 = ( -refLead + refRoot ) / refDenom    # normal solution
      referenceActivity2 = ( -refLead - refRoot ) / refDenom    # possible solution if |site.dissociatedCharge| >> |nativeIonCharge|
      print("Amphoteric site with asymmetric charge. Possible reference activities are: {}  or  {}".format(referenceActivity1,referenceActivity2));
      # adopt the positive solution as the reference
      # no way to distinguish them if they're both positive, so the first is taken then.
      if ( referenceActivity1 >= 0 ):
        referenceActivity = referenceActivity1;
      else:
        referenceActivity = referenceActivity2;
      
      if ( numpy.isfinite( Kvv ) ):
        referenceAssociation = 1 + referenceActivity/Kv * (1 + referenceActivity/Kvv);
      else:
        referenceAssociation = 1 + referenceActivity/Kv;
      
      # referenceBinding in units of Ns
      referenceBinding = referenceActivity/Kv * (1 + 2*referenceActivity/Kvv) / referenceAssociation;
      
    else:

      # common case: site.dissociatedCharge=0 or site.dissociatedCharge=-nativeIonCharge
      if (  Kdouble.size == 0 ):
	# Kvv=inf (asymmetric acidic)
        if site.dissociatedCharge==-nativeIonCharge:
          referenceActivity=numpy.inf  # make the inf explicit to avoid RuntimeWarning: "divide by zero encountered in double_scalars"
        else:
          referenceActivity = -Kv * site.dissociatedCharge/nativeIonCharge / ( 1 + site.dissociatedCharge/nativeIonCharge );
      else:
        if ( site.dissociatedCharge == 0 ):
	  # double binding on basic site (zero charge)
	  # reference activity is zero (referenceAssociation=1)
	  # "double binding" means other ions bind on top of the reference ion
	  # e.g. anions binding to -AH+
          referenceActivity = 0;
        else:
	  # amphoteric, symmetric site charge
          referenceActivity = sqrt( Kv * Kvv);
      if ( numpy.isfinite( Kvv ) ):
        referenceAssociation = 1 + referenceActivity/Kv * (1 + referenceActivity/Kvv);
      else:
        referenceAssociation = 1 + referenceActivity/Kv;
      # referenceBinding in units of Ns
      referenceBinding = -site.dissociatedCharge/nativeIonCharge;

    if nativeIonElectrolyteIndex >= 0:
      referenceNES =  nes[ nativeIonElectrolyteIndex ]
    else:
      if site.implicitNativeNES:
        referenceNES = site.implicitNativeNES
      else:
        referenceNES=0
    referenceToNeutralSurface = referenceBinding * referenceNES
    
    siteEnergyChemicalNonelectrostaticDontUse -= referenceToNeutralSurface;
     # Nonelectrostatic energy in kJ/mol, scale into microJ/m^2
    siteEnergyChemicalNonelectrostaticDontUse *= ( -site.siteDensityPerMSq  # 1/m^2, note minus sign
					   * 1000           # J/kJ
					   / Navo           # ions/mole
					   * 1e6            # microJ / J
                                            )
    energyChemicalNonelectrostaticDontUse += siteEnergyChemicalNonelectrostaticDontUse;
 
    ## double binding (amphoteric) particulars
    entropicCorrectionCoefficient = 0;
    energyTwoBindsite = 0;
    # mind went numb at this point. Revisit once chemisorption correction is processed.
    if (  Kdouble.size > 0 ):
      # calculate full C_{en}, taking am0=0 except for m=native
      referenceActivityAll = [0] * len( site.ionCompetitors );
      referenceActivityAll[ nativeIonCompetitorIndex ] = referenceActivity;
      
      ionCompetitorActivity = [0] * len(site.ionCompetitors)
      Kdiss = [0] * len(site.ionCompetitors)      
      for i in range(0,len(site.ionCompetitors)):
        ionCompetitorActivity[i] = surfacePartialActivity[site.ionCompetitors[i].electrolyteIndex]
        Kdiss[i] = 10**(-site.ionCompetitors[i].pKdissociation)
      
      # Δa = a-a0
      da = [ x-y for x,y in zip(ionCompetitorActivity, referenceActivityAll) ];
      A0 = referenceAssociation;
      A1=0;
      A2=0;
      B0 = [0] * len( site.ionCompetitors );
      B1 = [0] * len( site.ionCompetitors );
      B2 = [0] * len( site.ionCompetitors );
      for m in range(len(site.ionCompetitors)):
        if numpy.isfinite(Kdiss[m]):
          if numpy.isinf(Kdouble[m, nativeIonCompetitorIndex]):
            A1 += da[m] / Kdiss[m];
          else:
            A1 += da[m] / Kdiss[m] * ( 1 + referenceActivity / Kdouble[m, nativeIonCompetitorIndex] );
        sum_a = 0
        sum_da = 0
        for n in range(len( site.ionCompetitors )):
          sum_a += referenceActivityAll[n] * inv_K_double[m,n]
          sum_da += da[n] * inv_K_double[m,n]

        if pbconfig.useIncorrectDoubleBinding:
          sum_a += referenceActivityAll[m]*inv_K_double[m,m]
          sum_da += da[m]*inv_K_double[m,m]

        if numpy.isfinite(Kdiss[m]):
          B0[m] = referenceActivity * ( 1/Kdiss[m]  + sum_a )
          B1[m] = referenceActivity * sum_da + da[m] * ( 1/Kdiss[m] + sum_a )
        else:
          B0[m] = referenceActivity * sum_a
          B1[m] = referenceActivity * sum_da + da[m] * sum_a
        B2[m] = da[m] * sum_da

        if ( m==nativeIonCompetitorIndex ):
          for n in range(len(ionCompetitorActivity)):
            Kdmn_actual = Kdouble[m,n]
            if numpy.isfinite(Kdmn_actual):
              if numpy.isfinite(Kv):
                A1 += referenceActivity/Kv * da[n] / Kdmn_actual;
              if numpy.isfinite(Kdiss[m]):
                A2 += da[m]/Kdiss[m] * da[n] / Kdmn_actual;
	
      twoBindD = 0;
      
      entropicCorrectionCoefficient=1;
      for m in range(len(ionCompetitorActivity)):
        if numpy.isfinite(Kdiss[m]):
          num = da[m] * ( referenceActivityAll[m]*( A2*B1[m]-A1*B2[m] ) + da[m]*( A0*B2[m] - A2*B0[m] ) );
          denom = ( A2 * referenceActivityAll[m]**2 + da[m]**2*A0 -A1*referenceActivityAll[m]*da[m] );
          entropicCorrectionCoefficient -= num / (  2 * A2 * denom );
	
          Dnum = A2 * ( referenceActivityAll[m]*(A1*B1[m] + 2*A0*B2[m]) + da[m]*(A1*B0[m] - 2*A0*B1[m]) );
          Dnum += A0*A1*B2[m]*da[m] - ( (A1**2)*B2[m] + 2*(A2**2)*B0[m] )*referenceActivityAll[m];
          twoBindD += da[m] * Dnum / denom;
	
      twoBindD /= A2;

      # competitive double-binding takes an arctan or log form depending on the root
      # but calculate both for testing purposes
      
      if pbconfig.useIncorrectDoubleBinding:
        twoBindRootSq = 4 * associationFactor * A2 - A1**2;
      else:
        twoBindRootSq = 4 * A0 * A2 - A1**2;

      twoBindRoot = sqrt( twoBindRootSq );
      
      twoBindArctan = twoBindD * ( atan( (A1 + 2*A2) / twoBindRoot ) - atan(A1/twoBindRoot) ) / twoBindRoot;
      
      twoBindRootAlt = sqrt( -twoBindRootSq );
      twoBindLog = twoBindD/2 * ln( ( 2*A0 + A1 + twoBindRootAlt) / ( 2*A0 + A1 - twoBindRootAlt) ) / twoBindRootAlt;
      
      twoBindAdd = conditional(gt(twoBindRootSq,0),twoBindArctan,twoBindLog)
      energyTwoBindsite += twoBindAdd
      
      if ( len(ionCompetitorActivity) == 1 ):
	# if there is only one binding ion, then entropicCorrectionCoefficient=0 and twoBindD=0 exactly.
	# so only evaluate the case with competing ions
	# (or else machine numerics for a single ion will set D to "not quite 0", which is unhelpful)
        
	# but first report the calculated coefficients for truth-testing
        printf("Single binding ion, assigning exact coefficients\n");
        printf("entropicCorrectionCoefficient = 0  (reset from %g)\n", entropicCorrectionCoefficient );
        printf("twoBindD = 0 (reset from %g)\n", twoBindD);
        printf("energyTwoBindsite = 0 microJ/m^2 (reset from %g)\n", energyTwoBindsite * kB*temperature * chargeRegulatedSite.densityNumberPerAngSq * 1e20 * 1e6);
	
        entropicCorrectionCoefficient=0;
        energyTwoBindsite=0;
      
      # then take factor kT Ns  (J/m^2)
      energyTwoBindsite *= kB*temperature * site.siteDensityPerMSq
      
      # energyTwoBindsite in J/m^2, convert to microJ/m^2
      energyTwoBindsite   *= 1e6 ;        # microJ/J
    
    # === end competitive double-binding section
    
    energyTwoBind += energyTwoBindsite
    
    ## collect site entropic energy first in units of kT Ns
    #....slight difference in associationFactor, balances to difference of + vs -. Check assFac
    energyEntropicForThisSite = -ln(associationFactor);    # note minus sign: total association here, not unoccupied site fraction
    energyEntropicCorrectionForThisSite = ln(associationFactor) * entropicCorrectionCoefficient;    # note positive sign

    # subtract (add) native ion contribution relative to reference
    if nativeIonElectrolyteIndex >= 0:
      nativeSurfacePartialActivity = surfacePartialActivity[nativeIonElectrolyteIndex]
    else:
      nativeSurfacePartialActivity = site.implicitNativeIonBulkConcentration*exp(-getElectrostaticInteraction(nativeIonCharge,potential)/kT_kJmol)
    
    energyEntropicForThisSite += referenceBinding * ln( nativeSurfacePartialActivity / Kv );
    # and subtract the rest of the reference contribution
    if ( (site.dissociatedCharge == 0) or (nativeIonCharge == -site.dissociatedCharge) and numpy.isinf(Kvv) ):
      # avoid taking the log of 0 (qs=0)  or the limit of inf/(1+inf) (qs=-qv, Kvv=inf)
      referenceEntropicCorrection = 0;
      referenceEntropicCompetitiveCorrection = 0;
    else:  # general amphoteric case needs the reference correction
      referenceEntropicCorrection = ln(referenceAssociation) - referenceBinding * ln( referenceActivity / Kv );
      referenceEntropicCompetitiveCorrection = -ln(referenceAssociation) * entropicCorrectionCoefficient;
    energyEntropicForThisSite += referenceEntropicCorrection;
    energyEntropicCorrectionForThisSite += referenceEntropicCompetitiveCorrection;

    # then take factor kT Ns  (J/m^2)
    kT = sc.Boltzmann*temperature
    energyEntropicForThisSite *= kT * site.siteDensityPerMSq
    energyEntropicCorrectionForThisSite *= kT * site.siteDensityPerMSq

    # energyEntropicForThisSite in J/m^2, convert to microJ/m^2
    energyEntropicForThisSite  *= 1e6;       # microJ/J
    energyEntropicCorrectionForThisSite *= 1e6;
    energySiteEntropic += energyEntropicForThisSite;
    energySiteEntropicCorrection += energyEntropicCorrectionForThisSite;

    energyChemicalCompetitive = energySiteEntropicCorrection + energyTwoBind;
    
  return ( energySiteEntropic, energyChemicalCompetitive, energySiteEntropicCorrection, energyTwoBind, energyChemicalNonelectrostaticDontUse, boundChargeSI )



numEnergyComponents = 13  # energy components (not including total): electrostatic, entropic, nesEnergy, steric, chemisorption-electrostatic, chemisorption-siteEntropic, chemicalCompetitiveEnergy, chemSiteEntropicCorrectionEnergy, chemTwoBindEnergy,chemNonelectrostaticEnergyNotUsed, Casimir-Lifshitz


def getEnergyHeader():
  #header += "\tleftPotential(mV)"
  if opts.DerjaguinApproximation == 1:
    header = "L(nm)\tDerjaguinForceSphereSphere(microN/m)"
  else: # expect DerjaguinApproximation=2
    header = "L(nm)\tDerjaguinForceSpherePlate(microN/m)"
  header += "\ttotalEnergy(microJ/m^2)"
  header += "\telectrostatic"
  header += "\tentropic"
  header += "\tnesEnergy"
  header += "\tsteric"
  header += "\tcavity"
  header += "\tinterface"
  header += "\tchemElectrostaticEnergy"
  header += "\tchemSiteEntropicEnergy"
  header += "\tchemicalCompetitiveEnergy"
  header += "\tchemSiteEntropicCorrectionEnergy"
  header += "\tchemTwoBindEnergy"
  header += "\tchemNonelectrostaticEnergyNotUsed"
  header += "\tlifshitzEnergy"
  return header


# collect energy in microJ/m^2
# separation in nm
# potential in mV
# HamakerConstant in kT (default 0)
#
# energy returned as [ total, electrostatic, entropic, chemisorption-electrostatic, chemisorption-siteEntropic, Casimir-Lifshitz ]
def collectEnergy( separation, u_all, HamakerConstant=0 ):
  ulist = u_all.split()
  scaledPotential = ulist[0]
  scaledConcList = ulist[1:]
  potential = opts.potentialScale*scaledPotential # mV
  
  electrostaticEnergy = ( assemble(0.5*dot(-grad(scaledPotential),-grad(scaledPotential))*dx)
                          * sc.epsilon_0*EPS_0  # to SI units (J)
                          * 1e6    # microJ/J
                          * opts.potentialScale*opts.potentialScale  # solution scaling to (mV)^2
                          / ( 1000 * 1000 )  #  (V/mV)^2
                          / opts.lengthScale   # to 1/nm
                          * 1e9    # nm/m
                          )

  entropicEnergy = ( kT_kJmol*assemble(entropicDensity(scaledConcList,opts.bulkConcentration)*dx)
                     * 1e9   # microJ/kJ
                     * 1000  # L/m^3
                     * opts.lengthScale
                     * 1e-9  # m/nm
                     )

  nesEnergy = 0
  stericEnergy = 0
  cavityEnergy = 0
  interfaceEnergy = 0
  for i in range(len(scaledConcList)):
    ionConc = opts.bulkConcentration[i] * ( 2*exp(opts.concentrationScale[i]*scaledConcList[i]) - 1 )
    ionNESenergy = ( assemble(  ionConc * opts.ionNESinteraction[i] * dx )
                     * 1000  # L/m^3
                     * opts.lengthScale  # nm
                     * 1e-9  # m/nm
                     * 1e9   # microJ/kJ
                     )
    nesEnergy += ionNESenergy

    ionStericEnergy = ( assemble(  ionConc *  getStericEnergy(potential, opts.ionNESinteraction[i], opts.ionCharge[i], opts.stericEnergyFloor[i], opts.ionCavityEnergy[i], opts.ionInterfaceEnergy[i], opts.ionDCAenergy[i]) * dx )
                     * 1000  # L/m^3
                     * opts.lengthScale  # nm
                     * 1e-9  # m/nm
                     * 1e9   # microJ/kJ
                     )
    stericEnergy += ionStericEnergy
    
    ionCavEnergy = ( assemble(  ionConc * opts.ionCavityEnergy[i] * dx )
                     * 1000  # L/m^3
                     * opts.lengthScale  # nm
                     * 1e-9  # m/nm
                     * 1e9   # microJ/kJ
                     )
    cavityEnergy += ionCavEnergy 
    
    ionIntEnergy = ( assemble(  ionConc * opts.ionInterfaceEnergy[i] * dx )
                     * 1000  # L/m^3
                     * opts.lengthScale  # nm
                     * 1e-9  # m/nm
                     * 1e9   # microJ/kJ
                     )
    interfaceEnergy += ionIntEnergy 


    
  chemElectrostaticEnergy=0
  n = FacetNormal(u_all.function_space().mesh())
  for b in pbconfig.boundaryConditionList:
    if not b.type == BoundaryConditionType.CONSTANT_CHARGE:
      #  nuances for cons charge, separation surfaces
      chemElectrostaticEnergyOneBoundary = (
        -assemble(scaledPotential*dot(grad(scaledPotential),n)*opts.surfaceMeasure_ds(b.domainIndex))   # take care with negatives here
        * sc.epsilon_0*EPS_0  # SI units F/m = A^2s^4/(kgm^3) = C^2/(Nm^2) = J/(m V^2) = C/(m V) = C^2 /(m J)
        * 1e6    # microJ/J
        * opts.potentialScale*opts.potentialScale  # solution scaling to (mV)^2
        / ( 1000 * 1000 )  #  (V/mV)^2
        / opts.lengthScale   # to 1/nm
        * 1e9    # nm/m
      )
      chemElectrostaticEnergy += chemElectrostaticEnergyOneBoundary
  
  chemSiteEntropicEnergy =0     
  chemicalCompetitiveEnergy  =0
  chemSiteEntropicCorrectionEnergy =0
  chemTwoBindEnergy=0
  chemNonelectrostaticEnergyNotUsed =0
  for b in pbconfig.boundaryConditionList:
    if b.type == BoundaryConditionType.CHARGE_REGULATED:
      ( chemSiteEntropicEnergyForm, energyChemicalCompetitiveForm, energySiteEntropicCorrectionForm, energyTwoBindForm, energyChemicalNonelectrostaticDontUseForm, boundChargeSI ) = getFreeEnergyChemicalNonelectrostaticForSingleChargeRegulatedSurface(b.value,scaledConcList,potential,opts.ionNESinteraction)
      chemSiteEntropicEnergy           += assemble(chemSiteEntropicEnergyForm*opts.surfaceMeasure_ds(b.domainIndex))  
      chemNonelectrostaticEnergyNotUsed   += assemble(energyChemicalNonelectrostaticDontUseForm*opts.surfaceMeasure_ds(b.domainIndex))
      chemicalCompetitiveEnergy        += assemble(energyChemicalCompetitiveForm*opts.surfaceMeasure_ds(b.domainIndex))
      chemSiteEntropicCorrectionEnergy     += assemble(energySiteEntropicCorrectionForm*opts.surfaceMeasure_ds(b.domainIndex))
      chemTwoBindEnergy                   += assemble(energyTwoBindForm  *opts.surfaceMeasure_ds(b.domainIndex))                  
  
  lifshitzEnergy = ( -HamakerConstant / ( 12 * sc.pi * separation**2 )
                     * sc.Boltzmann * temperature
                     * 1e6    # microJ/J
                     * 1e18   # (nm/m)^2
                     )

  
  totalEnergy = electrostaticEnergy + entropicEnergy + nesEnergy + stericEnergy + cavityEnergy + interfaceEnergy #+ chemElectrostaticEnergy + chemSiteEntropicEnergy + chemicalCompetitiveEnergy + chemSiteEntropicCorrectionEnergy + chemTwoBindEnergy+ lifshitzEnergy
  return [ totalEnergy, electrostaticEnergy, entropicEnergy, nesEnergy, stericEnergy, cavityEnergy, interfaceEnergy, chemElectrostaticEnergy, chemSiteEntropicEnergy, chemicalCompetitiveEnergy, chemSiteEntropicCorrectionEnergy, chemTwoBindEnergy, chemNonelectrostaticEnergyNotUsed, lifshitzEnergy ]


pHList = [7]
#pHList = [7.15]
#pHList = [12.4]  # ref lysozyme
#pHList = [-1]
#pHList = numpy.unique( numpy.concatenate([numpy.arange(3,11+1)]) )
#pHList = pHList[pHList!=7]  # already done pH 7
###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################
#capacitance = (EPS_VAC* EPS_0)*np.cosh(qe*boundaryOuter.value*10**(-3)/(2*kB*temperature))
#print(EPS_VAC*EPS_0*50)
#pH = 7
#opts.pH = pH

#charge55 = getTotalElectronChargeDensity(55)
#print('Charge is {}'.format(charge55))

#'''

###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################



def getElectrodeChargeDensity(leftPotential,potentialDifference):
	global separation
	getBoundaryCondition(leftPotential,potentialDifference,opts.symmetricBoundary)
	opts.leftPotential = leftPotential
	
	updateElectrolyte()
	debyeLength = getDebyeLength(opts.bulkConcentration,opts.ionCharge)
	
	(u_allf,isConverged) = runPB(separation, debyeLength, doPrintSolution=True)
	
	if isConverged:
		allData = convertXDMF2OctavePB(separation, "-leftPotential{}-salt{}M".format(opts.leftPotential,opts.backgroundSaltConc))
	
	surfaceChargeDensity = sc.epsilon_0 * EPS_0 * 10**7 * allData[:,2]
	m = len(surfaceChargeDensity) - 1
	leftElectrodeChargeDensity = surfaceChargeDensity[0]
	RightElectrodeChargeDensity = surfaceChargeDensity[m]
	
	return (leftElectrodeChargeDensity, RightElectrodeChargeDensity)
	 


def getElectrodeCapacitanceDelPot(leftPotential,potentialDifference,delPotLeft): #calculates the differential capacitance for change in electrode potential (phi + delPhi)
	if (MPI.rank(mpi_comm) == rootMPI):
		print("Now calculating the differetntial capacitance")
	leftElectrodeChargeDensity, RightElectrodeChargeDensity = getElectrodeChargeDensity(leftPotential,potentialDifference)
	leftPotentialDelPot = leftPotential + delPotLeft
	getBoundaryCondition(leftPotentialDelPot,potentialDifference,opts.symmetricBoundary)
	lowerBound = potentialDifference - 10
	upperBound = potentialDifference + 10
	potentialDifferenceDelPot = seekChargeBalancedPotentialFromPD( lowerBound, upperBound)
	
	delPotRight = abs(delPotLeft + (potentialDifference - potentialDifferenceDelPot))
	leftElectrodeChargeDensityAtDelPot, RightElectrodeChargeDensityAtDelPot = getElectrodeChargeDensity(leftPotentialDelPot,potentialDifferenceDelPot)
	
	
	delPotRight = delPotLeft + (potentialDifference - potentialDifferenceDelPot)
	leftElectrodeChargeDensityAtDelPot, RightElectrodeChargeDensityAtDelPot = getElectrodeChargeDensity(leftPotentialDelPot,potentialDifferenceDelPot)
	
	leftElectrodeCapacitance = 100 * (leftElectrodeChargeDensityAtDelPot - leftElectrodeChargeDensity) * 1000 / delPotLeft  # 100 * is for converting F/m^2 to uF/cm^2
	
	RightElectrodeCapacitance = 100 * (RightElectrodeChargeDensity - RightElectrodeChargeDensityAtDelPot) * 1000 / delPotRight # 100 * is for converting F/m^2 to uF/cm^2
	TotalCapacitance = (leftElectrodeCapacitance * RightElectrodeCapacitance)/(leftElectrodeCapacitance + RightElectrodeCapacitance)
	opts.potentialDifference = potentialDifference
	
	return (leftElectrodeCapacitance , RightElectrodeCapacitance, TotalCapacitance)


def getElectrodeCapacitanceDelV(leftPotential,potentialDifference,DelV): #calculates the capacitance for change in potential difference (V + delV)
	
	leftElectrodeChargeDensity, RightElectrodeChargeDensity = getElectrodeChargeDensity(leftPotential,potentialDifference)
	potentialDifferenceDelPot = potentialDifference + DelV
	leftPotentialDelPot = seekleftPotential(potentialDifferenceDelPot)
	getBoundaryCondition(leftPotentialDelPot,potentialDifferenceDelPot,opts.symmetricBoundary)
	leftElectrodeChargeDensityAtDelPot, RightElectrodeChargeDensityAtDelPot = getElectrodeChargeDensity(leftPotentialDelPot,potentialDifferenceDelPot)
		
	delPotLeft = abs(leftPotentialDelPot-leftPotential)
	
	delPotRight = abs(delPotLeft - delPot)
		
	leftElectrodeChargeDensityAtDelPot, RightElectrodeChargeDensityAtDelPot = getElectrodeChargeDensity(leftPotentialDelPot,potentialDifferenceDelPot)
	
	leftElectrodeCapacitance = 100 * (leftElectrodeChargeDensityAtDelPot - leftElectrodeChargeDensity) * 1000 / delPotLeft  # 100 * is for converting F/m^2 to uF/cm^2
	
	RightElectrodeCapacitance = 100 * (RightElectrodeChargeDensity - RightElectrodeChargeDensityAtDelPot) * 1000 / delPotRight # 100 * is for converting F/m^2 to uF/cm^2
	TotalCapacitance = (leftElectrodeCapacitance * RightElectrodeCapacitance)/(leftElectrodeCapacitance + RightElectrodeCapacitance)
	opts.potentialDifference = potentialDifference
	
	return (leftElectrodeCapacitance , RightElectrodeCapacitance, TotalCapacitance)


def getStericLayerThickness(leftPotential,potentialDifference,backgroundIonList):
	rightPotential = leftPotential - potentialDifference
	cath , an = backgroundIonList
	conc_cap_left = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*an.gaussianRadiusAng**3)
	conc_cap_right = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*cath.gaussianRadiusAng**3)
	pot_threshold_left = -(sc.Boltzmann*temperature)/(-np.sign(leftPotential)*sc.e)*np.log(conc_cap_left/opts.backgroundIonConcentration)
	pot_threshold_right = -(sc.Boltzmann*temperature)/(-np.sign(rightPotential)*sc.e)*np.log(conc_cap_right/opts.backgroundIonConcentration)
	rho_cap_left = -np.sign(leftPotential)*sc.Avogadro*sc.e*conc_cap_left*1000
	rho_cap_right = -np.sign(rightPotential)*sc.Avogadro*sc.e*conc_cap_right*1000
	leftElectrodeChargeDensity, RightElectrodeChargeDensity = getElectrodeChargeDensity(leftPotential,potentialDifference)
	if(abs(leftPotential/1000) < abs(pot_threshold_left)):
		thickness_left = 0
	else:
		thickness_left = (-np.sign(leftPotential)/rho_cap_left)*(leftElectrodeChargeDensity-np.sqrt(leftElectrodeChargeDensity**2+2*EPS_0*sc.epsilon_0*rho_cap_left*(leftPotential/1000-pot_threshold_left)))
	if(abs(rightPotential/1000) < abs(pot_threshold_right)):
		thickness_right = 0
	else:
		thickness_right = (-np.sign(rightPotential)/rho_cap_right)*(RightElectrodeChargeDensity-np.sqrt(RightElectrodeChargeDensity**2+2*EPS_0*sc.epsilon_0*rho_cap_right*(rightPotential/1000-pot_threshold_right)))
	
	return (thickness_left,thickness_right)


def getTheoreticalFreeEnergy(leftPotential,potentialDifference,backgroundIonList):
	leftElectrodeChargeDensity, RightElectrodeChargeDensity = getElectrodeChargeDensity(leftPotential,potentialDifference)
	rightPotential = leftPotential - potentialDifference
	cath , an = backgroundIonList
	zLeft = -np.sign(leftPotential)
	zRight = -np.sign(rightPotential)
	thickness_left,thickness_right = getStericLayerThickness(leftPotential,potentialDifference,backgroundIonList)
	conc_cap_left = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*an.gaussianRadiusAng**3)
	conc_cap_right = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*cath.gaussianRadiusAng**3)
	rho_cap_left = -np.sign(leftPotential)*sc.Avogadro*sc.e*conc_cap_left*1000
	rho_cap_right = -np.sign(rightPotential)*sc.Avogadro*sc.e*conc_cap_right*1000
	pot_threshold_left = -(sc.Boltzmann*temperature)/(-np.sign(leftPotential)*sc.e)*np.log(conc_cap_left/opts.backgroundIonConcentration)
	pot_threshold_right = -(sc.Boltzmann*temperature)/(-np.sign(rightPotential)*sc.e)*np.log(conc_cap_right/opts.backgroundIonConcentration)
	mu_capLeft = pot_threshold_left*zLeft*sc.e
	mu_capRight = pot_threshold_right*zRight*sc.e
	
	epsilon = EPS_0*sc.epsilon_0
	F_en = 1000*sc.Avogadro*sc.Boltzmann*temperature*((conc_cap_left*np.log(conc_cap_left/opts.backgroundIonConcentration)-conc_cap_left+opts.backgroundIonConcentration)*thickness_left+(conc_cap_left*np.log(conc_cap_right/opts.backgroundIonConcentration)-conc_cap_right+opts.backgroundIonConcentration)*thickness_right)
	F_el = (0.5/epsilon)*((1/3)*rho_cap_left**2*thickness_left**3 -zLeft*leftElectrodeChargeDensity*rho_cap_left*thickness_left**2 +(zLeft*leftElectrodeChargeDensity)**2*thickness_left +(1/3)*rho_cap_right**2*thickness_right**3 -zRight*RightElectrodeChargeDensity*rho_cap_right*thickness_right**2 +(zRight*RightElectrodeChargeDensity)**2*thickness_right)
	F_st = (1000*sc.Avogadro*conc_cap_left*mu_capLeft - rho_cap_left*(leftPotential/1000-(rho_cap_left*thickness_left**2)/(6*epsilon)+(zLeft*leftElectrodeChargeDensity*thickness_left)/(2*epsilon)))*thickness_left + (1000*sc.Avogadro*conc_cap_right*mu_capRight - rho_cap_right*(rightPotential/1000-(rho_cap_right*thickness_right**2)/(6*epsilon)+(zRight*RightElectrodeChargeDensity*thickness_right)/(2*epsilon)))*thickness_right
	F = F_en + F_el + F_st
	#print(F,F_en,F_el,F_st)
	return (F,F_el,F_en,F_st,thickness_left,thickness_right)
	
	
def getCompositeDiffuseLayerCapacitance(leftPotential):
	RightPotential = leftPotential - opts.potentialDifference
	debyeLength = 1E-9*getDebyeLength(opts.bulkConcentration,opts.ionCharge) #1E-9 is to change it into nm
	conc_cap_left = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*an.gaussianRadiusAng**3)
	conc_cap_right = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*cath.gaussianRadiusAng**3)
	gammaLeft = 2*opts.backgroundIonConcentration/conc_cap_left
	gammaRight = 2*opts.backgroundIonConcentration/conc_cap_right
	uLeft = 0.001*abs(sc.e*leftPotential/(sc.Boltzmann*temperature)) #0.001 is to change it into V
	uRight = 0.001*abs(sc.e*RightPotential/(sc.Boltzmann*temperature)) #0.001 is to change it into V
	cap_0 = 100*sc.epsilon_0*EPS_0/debyeLength
	if(abs(leftPotential) < abs(sc.Boltzmann*temperature*np.log(conc_cap_left/opts.backgroundIonConcentration)/sc.e)):
		LeftCDL = 100*sc.epsilon_0*EPS_0 * np.cosh(uLeft)/debyeLength
	else:
		LeftCDL = cap_0*1/(np.sqrt(2*gammaLeft*(1-gammaLeft/2)**2+2*gammaLeft*abs(uLeft-np.log(2/gammaLeft))))
	if(abs(RightPotential)<abs(sc.Boltzmann*temperature*np.log(conc_cap_right/opts.backgroundIonConcentration)/sc.e)):
		RightCDL = 100*sc.epsilon_0*EPS_0 * np.cosh(uRight)/debyeLength
	else:
		RightCDL = cap_0*1/(np.sqrt(2*gammaRight*(1-gammaRight/2)**2+2*gammaRight*abs(uRight-np.log(2/gammaRight))))
	CDL = LeftCDL*RightCDL/(LeftCDL+RightCDL)
	
	return (LeftCDL, RightCDL, CDL)
	
	
def getKornyshevCapacitance(leftPotential):
	RightPotential = leftPotential - opts.potentialDifference
	debyeLength = 1E-9*getDebyeLength(opts.bulkConcentration,opts.ionCharge) #1E-9 is to change it into nm
	conc_cap_left = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*an.gaussianRadiusAng**3)
	conc_cap_right = 1E30/(1000*sc.Avogadro*np.pi*np.sqrt(np.pi)*cath.gaussianRadiusAng**3)
	gammaLeft = 2*opts.backgroundIonConcentration/conc_cap_left
	gammaRight = 2*opts.backgroundIonConcentration/conc_cap_right
	uLeft = 0.001*abs(sc.e*leftPotential/(sc.Boltzmann*temperature)) #0.001 is to change it into V
	uRight = 0.001*abs(sc.e*RightPotential/(sc.Boltzmann*temperature)) #0.001 is to change it into V
	cap_0 = 100*sc.epsilon_0*EPS_0/debyeLength
	LeftKornyshev = cap_0*(np.cosh(uLeft/2)/(1+2*gammaLeft*(np.sinh(uLeft/2))**2))*np.sqrt(2*gammaLeft*(np.sinh(uLeft/2))**2/(np.log(abs(1+2*gammaLeft*(np.sinh(uLeft/2))**2))))
	RightKornyshev = cap_0*(np.cosh(uRight/2)/(1+2*gammaRight*(np.sinh(uRight/2))**2))*np.sqrt(2*gammaRight*(np.sinh(uRight/2))**2/(np.log(abs(1+2*gammaRight*(np.sinh(uRight/2))**2))))
	Kornyshev = LeftKornyshev * RightKornyshev/(LeftKornyshev + RightKornyshev)
	
	return (LeftKornyshev, RightKornyshev, Kornyshev)
	
	
def getGouyChapmanCapacitance(leftPotential):
	RightPotential = leftPotential - opts.potentialDifference
	debyeLength = 1E-9*getDebyeLength(opts.bulkConcentration,opts.ionCharge) #1E-9 is to change it into nm
	uL = 0.001*abs(sc.e*leftPotential/(2*sc.Boltzmann*temperature)) #0.001 is to change it into V
	uR = 0.001*abs(sc.e*RightPotential/(2*sc.Boltzmann*temperature)) #0.001 is to change it into V
	LeftGCcap = 100*sc.epsilon_0*EPS_0 * np.cosh(uL)/debyeLength
	RightGCcap = 100*sc.epsilon_0*EPS_0 * np.cosh(uR)/debyeLength
	TotalGC = LeftGCcap*RightGCcap/(LeftGCcap+RightGCcap)
	return (LeftGCcap, RightGCcap, TotalGC)
	
###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################



cathion = [liIon] #,emtIon]#liIon]
anion = [pf6Ion]#ClIon, pf6Ion, bistriflimideIon, XIon]#pf6Ion, bf4Ion, clo4Ion, bro4Ion, io4Ion, ClIon]

for cath in cathion:
	for an in anion:
		
		symmetricBoundary = False
		opts.symmetricBoundary = symmetricBoundary
		backgroundIonList = [cath , an]
		opts.backgroundIonList = backgroundIonList
		backgroundIonConcentration = 1.2
		opts.backgroundIonConcentration = backgroundIonConcentration
		electrostaticOnly = True
		opts.electrostaticOnly = electrostaticOnly
		surfaceTension = 0 #41.1
		opts.surfaceTension = surfaceTension
		calculateCappacitance = True
		
		pHList = [7]
		for pH in pHList:
			opts.pH=pH
			updateElectrolyte()
			if (MPI.rank(mpi_comm) == rootMPI):
				print("{} ions in electrolyte:".format(len(opts.ionList)))
				print(list((ion.name,ion.charge,ion.concentration) for ion in opts.ionList))
			debyeLength=getDebyeLength(opts.bulkConcentration,opts.ionCharge)
				    
			#separationList=[2, 7, 190, 30*debyeLength]
			separationList=[20*debyeLength]
			#separationList=numpy.concatenate([[0.2,10],numpy.logspace(-0.2,numpy.log10(30*debyeLength),200)])##############check##################
			separationList=numpy.unique( numpy.array(separationList) )
	    	###############################################################################################################################################################################################################################################
	    	###############################################################################################################################################################################################################################################
	    	###############################################################################################################################################################################################################################################
	  
	  
			sIndex=0
			potDiffListIndex = 0
			leftPotentialList = [5]
			
			
			separation = 20*debyeLength
			delPot = 0.5
			DelV = 1
			potDiffList1 = [100]#np.linspace(20,600,30)#(20,100,5)
			potDiffList2 = []#np.linspace(20,100,5)#(20,500,25)#(150,550,9)
			potDiffList3 = []#np.linspace(150,600,10)#(550,2000,30)#(600,2600,11)
			potDiffList4 = []#np.linspace(700,2000,14)#[2700]
			# ~ potDiffList5 = np.linspace(520,580,4)
			# ~ potDiffList6 = np.linspace(620,680,4)
			potDiffList = np.concatenate((potDiffList1, potDiffList2, potDiffList3, potDiffList4))#, potDiffList5, potDiffList6))
			freeEnergyArray = numpy.zeros([len(potDiffList),8])
			energyArray = numpy.zeros([len(potDiffList),numEnergyComponents+5])
			capArray = numpy.zeros([len(potDiffList),14])    # 14 is len(capData) which is defined in the for loop below
			
			for pot in leftPotentialList:
				#opts.leftPotential = pot #+ delPot
				for potDiff in potDiffList:
					opts.potentialDifference = potDiff 
										
					if (potDiff != 0 and calculateCappacitance==False):
						capData = numpy.zeros(14) #14 is size of capArray
						opts.leftPotential = 0.5*opts.potentialDifference
						
					elif (potDiff == 0 and calculateCappacitance==False):
						capData = numpy.zeros(14) #14 is size of capArray
						opts.leftPotential = seekChargeBalancedPotential(opts.potentialDifference)
					else:
						opts.leftPotential = seekChargeBalancedPotential(opts.potentialDifference)
						leftElectrodeCapacitance, RightElectrodeCapacitance, TotalCapacitance = getElectrodeCapacitanceDelPot(opts.leftPotential,opts.potentialDifference,DelV)
						capData = [opts.leftPotential, opts.potentialDifference, leftElectrodeCapacitance, RightElectrodeCapacitance, TotalCapacitance]
						LeftGCcap,RightGCcap, TotalGC = getGouyChapmanCapacitance(opts.leftPotential)
						GCData = [LeftGCcap,RightGCcap, TotalGC]
						LeftCDL, RightCDL, CDL = getCompositeDiffuseLayerCapacitance(opts.leftPotential)
						CDLData = [LeftCDL, RightCDL, CDL]
						LeftKornyshev, RightKornyshev, Kornyshev = getKornyshevCapacitance(opts.leftPotential)
						KornyshevData = [LeftKornyshev, RightKornyshev, Kornyshev]
						capData = np.concatenate((capData, GCData, CDLData, KornyshevData))
						
					
					total,electrostatic,entropic,steric,leftThickness,rightThickness = getTheoreticalFreeEnergy(opts.leftPotential,opts.potentialDifference,opts.backgroundIonList)
					freeEnergyData = [opts.leftPotential, opts.potentialDifference,total,electrostatic,entropic,steric,1E10*leftThickness,1E10*rightThickness]
					freeEnergyArray[potDiffListIndex,:] = freeEnergyData
					capArray[potDiffListIndex,:] = capData
					potDiffListIndex += 1
					
					#opts.leftPotential = seekleftPotential
					getBoundaryCondition(opts.leftPotential,opts.potentialDifference,opts.symmetricBoundary)
				
				
					updateElectrolyte()
					debyeLength = getDebyeLength(opts.bulkConcentration,opts.ionCharge)
					separation = 20*debyeLength
					
					if(separation>0):
						(ufinal,isConverged) = runPB(separation,debyeLength, doPrintSolution=False)
						if isConverged:
							energy = collectEnergy(separation, ufinal, opts.HamakerConstant_kT)
							energyArray[sIndex,:] = [ separation, opts.leftPotential, opts.potentialDifference, opts.DerjaguinApproximation*scipy.pi*energy[0] ] + energy
							# DerjaguinApproximation: 1 = sphere-sphere, 2 - sphere-plate or cylinder-cylinder
							if (MPI.rank(mpi_comm) == rootMPI):
								print("energy (microJ/m^2) at {}nm = ".format(separation), energy)
							previous_u_allf = ufinal.copy(deepcopy=True)
							# ~ dispersionB = [cath.dispersionB[0],an.dispersionB[0]]
							# ~ vacuumDispersionB = [cath.vacuumDispersionB[0],an.vacuumDispersionB[0]]
							
							# ~ for i in range(len(opts.cavityRadius)):
								# ~ plotEnergy(0,separation,opts.cavityRadius[i],dispersionB[i],vacuumDispersionB[i],opts.gaussianRadius[i],i,debyeLength,opts.leftPotential)
						sIndex += 1
					
				updateElectrolyte()
				capArray=capArray[~(numpy.isclose(capArray,0)).all(1)]
				energyArray=energyArray[~(numpy.isclose(energyArray,0)).all(1)]
				ionsDirectory = opts.backgroundIonList[0].name + opts.backgroundIonList[1].name
				if energyArray.size > 0: # don't print broken configurations with no solutions!
					if opts.potentialDifference == 0:
						directory = ionsDirectory + '/DischargedState'
						try:
							if not os.path.isdir(directory):
								os.makedirs(directory)
						except OSError:
							print('Directory Already Exist 2 ' + directory)
						
					else:
						directory = ionsDirectory + '/ChargedState'
						try:
							if not os.path.isdir(directory):
								os.makedirs(directory)
						except OSError:
							print('Directory Already Exist 2 ' + directory)
						if calculateCappacitance:
							capFileName = '{}/ElectrodeCapacitance-leftPot{}-potDiff{}.dat'.format(directory,opts.leftPotential,opts.potentialDifference)
							capHeader="leftPotential(mV)\tPotDifference(mV)\tleftCapacitance(uF/cm^2)\tRightCapacitance(uF/cm^2)\tTotalCapacitance(uF/cm^2)\tleftGC(uF/cm^2))\tRightGC(uF/cm^2)\tTotalGC(uF/cm^2)\tleftCDL(uF/cm^2)\tRightCDL(uF/cm^2)\tTotalCDL(uF/cm^2)\tleftKornyshev(uF/cm^2)\tRightKornyshev(uF/cm^2)\tTotalKornyshev(uF/cm^2)"
							numpy.savetxt(capFileName, capArray, fmt="%.10g", delimiter="\t", newline='\n', header=capHeader)
							
					energyFileName='{}/energy-leftPotential{}-potDiff{}.dat'.format(directory,opts.leftPotential,opts.potentialDifference)
					energyheader = "L(nm)\tleftPotential(mV)\tPotDifference(mV)" + getEnergyHeader()
					numpy.savetxt(energyFileName, energyArray, fmt="%.10g", delimiter="\t", newline='\n', header=energyheader)
					freeEnergyFileName = '{}/{}-freeEnergyApproximation-leftPot{}-potDiff{}.dat'.format(directory,ionsDirectory,opts.leftPotential,opts.potentialDifference)
					freeEnergyHeader = "leftPotential(mV)\tPotDifference(mV)\tTotal(J/m^2)\tElectrostatic(J/m^2)\tEntropic(J/m^2)\tSteric(J/m^2)\tleftThickness(Ang)\trightThickness(Ang)"
					numpy.savetxt(freeEnergyFileName, freeEnergyArray, fmt="%.10g", delimiter="\t", newline='\n', header=freeEnergyHeader)
								
	    
#numpy.savetxt('energy.dat', energyArray, header=getEnergyHeader(), encoding='utf-8')  # numpy 1.14 not yet available ;(
# ugly hack, numpy does not freaking well handle unicode ;(
# https://stackoverflow.com/questions/41528192/specifying-encoding-using-numpy-loadtxt-savetxt
# https://github.com/numpy/numpy/pull/8644
#with open('energy.dat', 'wb') as f:
#    numpy.savetxt(f, energyArray, header=bytes(getEnergyHeader().encode('utf-8')))
#'''


endTime = time.perf_counter()
runTime = endTime - startTime

if (MPI.rank(mpi_comm) == rootMPI):
  print("Total run time " + str(runTime) + " sec on " + str(MPI.size(mpi_comm)) + " processes.   " + str(MPI.size(mpi_comm)) + "\t" + str(runTime) )

#'''

