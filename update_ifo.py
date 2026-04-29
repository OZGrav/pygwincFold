import numpy as np
import copy


def update_FMsus(ifo, mass, length=1):
    n = len(ifo.FoldingMirrorSuspension.Stage)
    masses = np.zeros(n)
    for i in range(n):
        masses[i] = ifo.FoldingMirrorSuspension.Stage[i].Mass
    masses_new = copy.deepcopy(masses)
    masses_new[0] = mass 
    masses_1 = np.cumsum(masses)
    masses_2 = np.cumsum(masses_new)
    mass_ratio = masses_2/masses_1
    for i in range(n):        
        ifo.FoldingMirrorSuspension.Stage[i].Dilution = np.nan
        ifo.FoldingMirrorSuspension.Stage[i].Blade = ifo.FoldingMirrorSuspension.Stage[i].Blade * np.sqrt(mass_ratio[i])
        ifo.FoldingMirrorSuspension.Stage[i].K = ifo.FoldingMirrorSuspension.Stage[i].NWires*ifo.FoldingMirrorSuspension.Stage[i].K * 0.25*np.sqrt(mass_ratio[i]**3)
        ifo.FoldingMirrorSuspension.Stage[i].Length = ifo.FoldingMirrorSuspension.Stage[i].Length * length
        if i != 0:
            ifo.FoldingMirrorSuspension.Stage[i].WireRadius = ifo.FoldingMirrorSuspension.Stage[i].WireRadius * np.sqrt(mass_ratio[i])
        if ifo.FoldingMirrorSuspension.FiberType == "Tapered":
            ifo.FoldingMirrorSuspension.Fiber.Radius = np.sqrt( mass*9.8/4/734858421/np.pi)
            ifo.FoldingMirrorSuspension.Fiber.EndRadius = np.sqrt((mass * 9.8 * ifo.FoldingMirrorSuspension.Silica.dlnEdT)/(ifo.FoldingMirrorSuspension.Stage[0].NWires* np.pi* ifo.FoldingMirrorSuspension.Silica.Alpha * ifo.FoldingMirrorSuspension.Silica.Y))
        elif ifo.FoldingMirrorSuspension.FiberType == "Ribbon":
            ifo.FoldingMirrorSuspension.Ribbon.Thickness = ifo.FoldingMirrorSuspension.Ribbon.Thickness * np.sqrt(mass_ratio[0])
            ifo.FoldingMirrorsuspension.Ribbon.Width = ifo.FoldingMirrorSuspension.Ribbon.Thickness * 10
        else:
            print("Error: Fiber type is neither tapered nor ribbon")
    ifo.FoldingMirrorSuspension.Stage[0].Mass = mass
    ifo.Materials.FoldingMirrorMassThickness = np.pi * ifo.Materials.FoldingMirrorMassRadius**2 * ifo.Materials.FoldingMirrorSubstrate.MassDensity / mass
    

def update_sus(ifo, mass, length=1):
    n = len(ifo.Suspension.Stage)
    masses = np.zeros(n)
    for i in range(n):
        masses[i] = ifo.Suspension.Stage[i].Mass
    masses_new = copy.deepcopy(masses)
    masses_new[0] = mass 
    masses_1 = np.cumsum(masses)
    masses_2 = np.cumsum(masses_new)
    mass_ratio = masses_2/masses_1
    for i in range(n):        
        ifo.Suspension.Stage[i].Dilution = np.nan
        ifo.Suspension.Stage[i].Blade = ifo.Suspension.Stage[i].Blade * np.sqrt(mass_ratio[i])
        ifo.Suspension.Stage[i].K = ifo.Suspension.Stage[i].NWires*ifo.Suspension.Stage[i].K * 0.25*np.sqrt(mass_ratio[i]**3)
        ifo.Suspension.Stage[i].Length = ifo.Suspension.Stage[i].Length * length
        if i != 0:
            ifo.Suspension.Stage[i].WireRadius = ifo.Suspension.Stage[i].WireRadius * np.sqrt(mass_ratio[i])
        if ifo.Suspension.FiberType == "Tapered":
            ifo.Suspension.Fiber.Radius = np.sqrt( mass*9.8/4/734858421/np.pi)
            ifo.Suspension.Fiber.EndRadius = np.sqrt((mass * 9.8 * ifo.Suspension.Silica.dlnEdT)/(ifo.Suspension.Stage[0].NWires* np.pi* ifo.Suspension.Silica.Alpha * ifo.Suspension.Silica.Y))
        elif ifo.Suspension.FiberType == "Ribbon":
            ifo.Suspension.Ribbon.Thickness = ifo.Suspension.Ribbon.Thickness * np.sqrt(mass_ratio[0])
            ifo.Suspension.Ribbon.Width = ifo.Suspension.Ribbon.Thickness * 10
        else:
            print("Error: Fiber type is neither tapered nor ribbon")
    ifo.Suspension.Stage[0].Mass = mass
    ifo.Materials.MassThickness = np.pi * ifo.Materials.MassRadius**2 * ifo.Materials.Substrate.MassDensity / mass
    