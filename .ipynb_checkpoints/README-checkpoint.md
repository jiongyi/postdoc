# README

This repository contains all the code I've developed in my postdoc in the Mullins lab at UCSF.

## Contents summary

### Actin assembly

- Code to segment beads and comet tails and quantify bead motility data
- Nucleation-promoting factor densities and network-growth rates from WAVE and N-WASP bead motility data

### DNA damage

- Code to segment immunofluorescent DNA repair factors in U2OS cells
- Quantification of DNA repair factor clusters and their colocalization in U2OS cells treated by cesium irradiation or bleocin
- Code to fit binding equations to fluorescence anisotropy data
- Binding affinities estimated for Arp2/3 complex and Arpc1 subunit to DNA

### Simulation

- Monte Carlo simulation for Weichsel-Schwarz model from PNAS 2010
- Monte Carlo simulation for Mueller-et-al-Sixt model from Cell 2017
- Monte Carlo simulation of homebrewed model (tan-mullins)

#### Usage

In all nucleation modules, simulations are implemented in the Network class.

Import nucleation module and create Network instance.

~~~
from nucleation import *
actin_network_obj = Network()
~~~

Run the simulation.

~~~
actin_network_obj.simulate()
~~~

Display network.

~~~
actin_network_obj.display_network()
~~~
