
# Python Gravitational Wave Interferometer Noise Calculator for Folded Interferometers

Code that matters is in gwincFold, I haven't put any effort into making the CLI work so please just use it in a python script or a jupyter notebook.

The IFOs that are currently working are:
- APlusFold
- APlusFoldAlGaAs
- APlusFoldAlGaAsSilicon

There actual params are not necessarily reliable. APlusFold has the folding mirror with a FS substrate and suspension with tantala/silica coatings, APlusFoldAlGaAs has the folding mirror coating switched so that the loss angle is close to AlGaAs (based on the A# AlGaAs models in the LIGO post-05 gitlab made by kevin kuns) and the APlusFoldAlGaAsSilicon has the folding mirror substrate and suspension replaced with 123K silicon (material params taken from gwinc CE2Silicon.yaml). The coating thickness for the last yaml I also had to borrow from the CE2Silicon model since it broke something but this needs to be fixed at some point.

# Task List

- AlGaAs coating parameters
- Silicon Suspension Parameters
  
