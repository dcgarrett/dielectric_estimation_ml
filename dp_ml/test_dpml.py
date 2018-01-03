# Run using: python -m unittest test_dpml.py

import dp_ml
import unittest
import numpy as np
import os
from .config import *

class TestDPML(unittest.TestCase):
	def test_add(self):
		result = dp_ml.add(10,5)
		self.assertEqual(result,15)

	def test_params_from_filename(self):
		test_filename = 'eps20_sig5_50mm.xls'
		eps, sig, dist = dp_ml.get_params_from_filename(test_filename)

		self.assertEqual(eps, 20)
		self.assertEqual(sig, 5)
		self.assertEqual(dist, 50)

	def test_realimag_to_magphase(self):
		
		mag, phase = dp_ml.realimag_to_magphase(1,1)
		self.assertEqual(mag, np.sqrt(2))
		self.assertEqual(phase, 45*np.pi/180)

	def test mag2db(self):
		result = dp_ml.mag2db(10)
		self.assertEqual(result, 20)

	def 


if __name__ == '__main__':
	unittest.main()
