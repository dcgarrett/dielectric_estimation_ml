.. dp_ml documentation master file, created by
   sphinx-quickstart on Tue Dec 12 09:58:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tutorial
=================================

.. image:: /figures/dp_ml_logo.png
   :align: center

.. toctree::
   :maxdepth: 2

This guide will run you through a typical workflow for the dp_ml package.
We will first generate/download some data, then store it in a convenient form.
Next, we will generate models from the simulated data, and then perform some property estimation in both simulation and measurement.

All of the steps followed below are given in a Jupyter Notebook for convenient testing. 

data generation
-----------------

You have two options for obtaining training data:
   1. simulate your own
   2. download the from the growing database online

model generation
-----------------

Now that we have our training data, we can generate models to be used for prediction.

.. code-block:: py

	import dp_ml

system calibration
-------------------

property estimation
--------------------



