.. dp_ml documentation master file, created by
   sphinx-quickstart on Tue Dec 12 09:58:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

converting between simulation and measurement
==============================================

This is a crucial step to the utility of applying models from simulation to measurement.
One of our assumptions in creating this toolbox is that there exists some function which can map our simulated data to our measured data, and vice-versa.
Here we will attempt to do this.

To examine the difference between simulations using the circular waveguide and measurements with the Nahanni, several materials were assessed at many separation distances in both scenarios.
This allows for the determination of a transformation which allows us to compare simulated and measured results.
Examined materials include air, canola oil, and tap water.
We are basically searching for a transfer function which allows us to "convert" between simulation and measurement.

The general process is as follows:

1. Apply antenna calibration in measurement to move the reference planes to the antenna apertures.
2. Add the response of the waveguides to the calibrated measured data.
3. Find some transfer function which is reasonable for the variety of materials assessed.

1. calibrate measurements
````````````````````````````

Calibration coefficients from measurement :math:`O_{meas}` and :math:`P_{meas}` are first found using Gate-Reflect-Line.

The antenna response is then 

2. add waveguide response
````````````````````````````

3. find transfer function
````````````````````````````

