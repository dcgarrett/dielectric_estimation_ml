.. dp_ml documentation master file, created by
   sphinx-quickstart on Tue Dec 12 09:58:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

background
=================================

.. image:: figures/dp_ml_logo.png
   :align: center

The dp_ml package uses a few forms of machine learning to develop models for property estimation.
Namely, neural networks and elastic net regression.
We also use different algorithms for processing the raw data prior to generating the models, such as wavelet decomposition and peaks analysis.

This document will describe some of the underlying math.
It may be best to read this document prior to the Tutorial document, in order to better understand the steps it takes.

the problem
------------

Let us simply define the problem we are trying to solve:
	- Antenna measurements allow for non-invasive interrogation of tissues by propagating electric fields within them
	- Dielectric properties cannot be directly measured. Instead we measure parameters such as the reflected and transmitted signals between the two antennas. These are called the Scattering Parameters (S-Parameters), and can be considered in both the time and frequency domain.
	- We now have an inverse problem. The dielectric properties affect the S-Parameters, but we don't know what they were to begin with. How can we go from the S-Parameters back to the dielectric properties?

The solution that we use in this method is that **in simulation, we do know what the properties are**. 
This means that for every S-Parameter, we can know exactly what dielectric properties this corresponds to.
Machine learning comes into play by figuring out how to use all of the data that we simulate to create models capable of estimating properties in the future, particularly from S-parameters from measurements.
The basic idea is to create a training set --- that is, a large set of data with known properties --- in order to generate these models which can perform predictions from future S-parameters when dielectric properties are not known.


simulations
-------------------

We use finite difference time domain (FDTD) simulations to generate our training data.

The antenna developed in our group - dubbed the Nahanni - has very high performance for biomedical applications, with large bandwidth (1.5-12 GHz), matching with human tissues, and isolation from exterior signals. 
However, its large and complex geometry causes long simulation time. 
To overcome this, two simplified models are considered. 
First, the antenna is modelled as a cylindrical waveguide with similar dimensions and filling materials. 
The second approach involves modelling the aperture of the realistic antenna using a Huygens surface equivalence source. 
To do this, an initial simulation is performed where fields incident upon the antenna aperture are recorded. 
In future simulations, these incident fields are then recreated without needing to simulate the body of the antenna.

Our training configuration consists of two cylindrical waveguides separated by the tissue under test.
If desired, you can simulate a different configuration (see the Tutorial section).
We simulate over a broad band of frequency: 2-12 GHz.
The resulting scattering parameters can be represented in both the time domain and frequency domain.

.. figure:: figures/WaveguideDiagram.png
   :align: center
   :scale: 60

   Our simulation configuration, as performed using Sim4Life

In both cases, the dielectric properties of the tissue are swept in a grid as:

$$ \\epsilon = 2:2:68 $$
$$ \\sigma = 0:1:10 $$

This range of values represents expected properties in biological tissues. 
Data is extracted in the form of magnitude/phase of reflected (S11) and transmitted (S21) signals, at 5000 frequency points between 2-12 GHz.




electromagnetic theory
----------------------

The wave propagation can be represented as a uniform plane wave.

$$ \\alpha = \\omega \\sqrt{\\frac{\\mu \\epsilon}{2} \\bigg[ \\sqrt{1 + [\\frac{\\sigma}{\\omega \\epsilon}]^2 } -1 \\bigg] } $$
$$ \\beta = \\omega \\sqrt{\\frac{\\mu \\epsilon}{2} \\bigg[ \\sqrt{1 + [\\frac{\\sigma}{\\omega \\epsilon}]^2 } + 1 \\bigg] } $$

The reflections can be determined by the intrinsic impedance of each medium, where:

$$ \\eta_{tiss} = \\sqrt{\\frac{j \\omega \\mu}{\\sigma + j \\omega \\epsilon}} $$

Next, the transmission coefficient at each interface can be found as:

$$ T = \\frac{2 \\eta_2}{\\eta_2 + \\eta_1} $$

data preprocessing
------------------

Different forms of processing can be applied to the data prior to model generation.

peak detection
````````````````
The simplest may be simply determining the peak the transmitted time domain pulse.
This is an efficient form of data storage, as a large time domain signal (~5000 samples) can be reduced to a simple two-element array (magnitude and time of the maxima).

wavelet analysis
`````````````````

A conventional inner product for two continuous functions which meet the :math:`L^2` criteria is:

$$ \\langle f(t), w(t) \\rangle = \\int_a^b f(t)w(t) dt $$

where :math:`f(t)` is the signal of interest (e.g. a specific time-domain transmission signal for given properties), and :math:`w(t)` is a mother wavelet.

Since digital signals are inherently discrete, this can also be expressed as:

$$ \\langle f, w \\rangle = \\sum_{n=0}^{n=N} f[n] w[n] $$

where :math:`N` is the number of time steps, and :math:`n` is the current time sample.

This can be done conveniently in python using 

.. code-block:: py

    import numpy as np
    np.inner(f,w)
    
The trick for us will be to generate the mother wavelet, meeting the orthonormality requirements, and ensuring equal number of samples.

A couple options for the mother wavelet:

    - Use the time domain signal from the Thru simulation
    - Use the time domain signal from 10 mm; eps = 25; sig = 0 simulation
    - Use a conventional wavelet such as Mexican hat

frequency domain analysis
``````````````````````````



machine learning
-----------------

This modelling problem can be described by the following:
	- Continuous regression (estimate permittivity and conductivity)
	- ∼ 2-12 features (e.g. Mag/phase of S-parameters and their feature expansions)
	- Training data on order of 1000s of samples

Two techniques are found to be suitable candidates and will be discussed below: elastic net regression, and neural network regression. 
All data processing is done through the open-source toolbox *tensorflow*.

neural networks
```````````````
The basic idea of a neural network is first described. 


In our method,

.. figure:: figures/ann_diagram.png
   :align: center
   :scale: 50 %


Neural networks loosely model the functionality of biological neurons, whereby neurons respond to an input stimulus with some corresponding output. 
The process of “learning” is simply determining the set of input weights and biases which produce the least error at the output with the training data. 
A single hidden layer is sufficient to model any arbitrary function. 
However, the advantage of multiple hidden layers is to reduce incidence of local minima, and to create more abstract models. 
Any model with two or more hidden layers is referred to as “deep learning”.

In the case of dielectric property estimation, the input layer consists of the extracted features: magnitude and phase data of the reflected and transmitted signals. 
The differentiated phase is again considered since wrapped phase provides little information to the network. 
There are two continuous outputs, permittivity and conductivity. 
The neural network configuration is shown in Figure X. 
Note that a model is generated for each frequency point.

elastic net regression
```````````````````````

Elastic net regression is a flexible method which combines strengths from several other models: ordinary least squares (OLS) regression, ridge regression, and LASSO regression. 
One might wonder what the issue is with ordinary least squares regression. 
In fact, the Gauss-Markov theorem states that of all unbiased estimators, OLS has the minimum variance in coefficient estimation. 
It is interesting to question, though, whether this is the best method purely in terms of mean-squared error (MSE). 
For instance, if we introduce a biased estimator - i.e. the mean of the estimated coefficients is offset from the true mean - we can reduce variance. 
This is shown in the figure below:

**figure**

So even though the estimator is biased, the mean squared error can actually be less than the OLS case! 
Ridge and LASSO achieve this by introducing penalty terms for the complexity of the model, where the complexity is determined by the l2 and l1 norms of the model coefficients, respectively. 
Elastic net combines these two penalty terms, as shown below:


.. math::

	\hat{\beta}^{OLS} &= \textrm{argmin}||y-X\beta||_2^2 \\
	\hat{\beta}^{ridge} &= \textrm{argmin}||y-X\beta||_2^2 + \color{red} \lambda_2||\beta||_2^2 \\ 
	\hat{\beta}^{lasso} &= \textrm{argmin}||y-X\beta||_2^2 + \color{blue} \lambda_1||\beta||_1  \\
	\hat{\beta}^{elastic} &= \textrm{argmin}||y-X\beta||_2^2 + \color{red} \lambda_2||\beta||_2^2 \color{black} + \color{blue} \lambda_1||\beta||_1 \\ 


