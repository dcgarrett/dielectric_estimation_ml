.. dp_ml documentation master file, created by
   sphinx-quickstart on Tue Dec 12 09:58:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

simulation
=================================

.. image:: figures/dp_ml_logo.png
   :align: center


We use finite difference time domain (FDTD) simulations to generate our training data.
This allows us to model the complex interactions between the antenna and the tissue.
The antenna developed in our group - dubbed the Nahanni - has very high performance for biomedical applications, with large bandwidth (1.5-12 GHz), matching with human tissues, and isolation from exterior signals.
However, its large and complex geometry causes long simulation time.
To overcome this, we model it as a cylindrical waveguide with similar dimensions and filling material.
These simulations can be performed at a fraction of the time, facilitating the generation of training data.
However, we then require a method of interpreting measured and simulated data with the full antenna model using these simplified simulations.


We consider three increasingly complex models of human tissues:

1. Homogenous tissues
2. Layered tissues
3. Realistic human models


1. homogeneous tissues
````````````````````````

We first simulate using a homogeneous model.
Our training configuration consists of two cylindrical waveguides separated by the tissue under test.
If desired, you can simulate a different configuration (see the Tutorial section).
We simulate over a broad band of frequency: 2-12 GHz.

.. figure:: figures/WaveguideDiagram.png
   :align: center
   :scale: 80

The dielectric properties of the tissue are swept in a grid as:

$$ \\epsilon = 2:2:68 $$
$$ \\sigma = 0:1:10 $$

This range of values represents expected properties in biological tissues.
The tissue thickness is swept as:

$$ d = 10:10:70 \\textrm{mm} $$

This also represents the range of expected thicknesses, particularly at the arm.

Resulting 2-port S-Parameters are extracted as complex values at 5000 frequency points between 2-12 GHz.
We could consider this as some 3-dimensional space where we modulate the dielectric properties and thickness, and extract the complex S-parameters :math:`\mathbf{X}`.

.. figure:: figures/tex/dp_figs-16.png
   :align: center
   :scale: 60

2. layered tissues
````````````````````


The challenge in simulating layered tissues is the high dimensionality.
That is, there are many more possible configurations of the tissue, where if they were all accounted for would result in exponentially more simulations.
To restrict this dimensionality, we consider cases similar to the forearm where there is skin on the exterior, with fat below the surface of the skin, filled with muscle.
Skin is treated with uniform thickness and properties.
Fat thickness is adjusted within expected ranges (2-6 mm) but with constant dielectric properties.

Similar simulations are performed, but we only manipulate the properties of these tissues within narrower ranges around the expected properties.
The thickness of muscle is also altered within expected ranges, resulting in arm thicknesses of 4-8 cm.

This model is shown below.

.. figure:: figures/LayeredTissue.png
   :align: center
   :scale: 60

Here we have added one dimension to our space of input parameters to the simulation :math:`\mathcal{D}_{in,layer}`.

.. figure:: figures/tex/dp_figs-17.png
   :align: center
   :scale: 60

3. realistic human models
`````````````````````````


For a highly representative model of human tissues, we can use MRI-derived models from the IT'IS Foundation's Virtual Population.
Tissues are segmented, and associated properties are assigned.
Depending on age and gender of the subject, several models can be used. 
For instance, we use 84 and 34 year-old males and a 26 year-old female.

The issue again here is the dimensionality of the simulation space; there are very many different locations we could assess, and many different tissues which could change depending on the individual. 
We can restrict this by limiting simulations to certain locations.
For instance, we assess the forearm for hydration assessment, where antennas are to be positioned at the midpoint between the wrist and the elbow.

.. figure:: figures/sim4life_nahanni.png
   :align: center
   :scale: 40


We start with the model as a baseline representation of this demographic.
The features which are modified may depend on the application.
In short-term hydration assessment, we expect bone and fat properties to remain constant, while muscle properties may change.
Subcutaneous fat thickness may vary by subject, so we consider this as a dimension. 
Each subject's total arm thickness may vary, which we can represent by modifying the muscle thickness.
This gives us a few dimensions to operate in.
We make the model pseudo-planar by creating flat skin and fat surfaces at the aperture of either antenna.
This emulates the slight physical compression which occurs in measurement.

.. figure:: figures/humanmodel.png
   :align: center
   :scale: 25

We now add one more dimension to our search space, where the demographic of the measured subject :math:`A` will determine the appropriate simulation model to use. 
This will primarily change the bone and fat location and proportion within the arm.
Note that we currently only have access to three of these models.
As more segmented models are generated, this will become a better age- and sex-matched representation of each volunteer.

.. figure:: figures/tex/dp_figs-18.png
   :align: center
   :scale: 60


data storage
````````````````

All extracted S-parameters are stored in hierarchical data format (HDF5).
This allows for compressed storage but in an organized manner.




