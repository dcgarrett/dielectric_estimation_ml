# Some math functions which aren't as easily accessible through numpy
	# David Garrett - October 2017

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.fft as FFT
from scipy import signal



# Pulse Parameters
tao=62.5e-12;
to=4*tao;


def inverseczt_charlotte(freq,s11):
	# TSAR pulse parameters:
	#dt_tsar=2e-12;   #Time Step
	T_tsar = 10e-9 
	dt_tsar = T_tsar / len(freq) # 1e-11 for 1000, 2e-12 for 5000;

	#T_tsar= 10e-9;     #Time Length
	t_tsar=np.arange(0,T_tsar,dt_tsar)   #Time Vector
	t = t_tsar

	N = len(t) # 5000
	dt = t[1] - t[0]
	df = freq[1] - freq[0]
	fmin = min(freq)
	M = len(freq) # 1000
	W = np.exp(-1j*2*np.pi*dt*df) # scalar
	A = np.exp(1j*2*np.pi*fmin*dt) # scalar

	tpulse = TSAR_handle(t) # 5000

	fpulse = chirpz(tpulse, A,W,M) # 1000
	#fpulse = chirpz3(tpulse, A,W,M) # 1000

	fpulse = fpulse/max(np.absolute(fpulse))
	if len(fpulse) == len(s11):
		fsig = fpulse*s11
	else:
		fpulse = np.append(fpulse, 0)
		fsig = fpulse*s11

	A = np.exp(1j*2*np.pi*t[0]*df) # scalar
	bttczt = np.conj(chirpz(np.conj(fsig),A,W,N))*df*np.sqrt(2) # should be 5001 points
	#bttczt = np.conj(chirpz3(np.conj(fsig),A,W,N))*df*np.sqrt(2) # should be 5001 points
	phaseshift = np.exp(1j*2*np.pi*fmin*np.arange(0,N)*dt)
	complexsig = phaseshift*bttczt
	
	# SHOULD THIS BE ABSOLUTE INSTEAD OF REAL??? TRYING THIS NOW
	tsig_real = np.real(complexsig)
	tsig = np.absolute(complexsig)
	
	return t, tsig, tsig_real

def inverseczt_tukey(freq,s11,alpha=0.01):
	# TSAR pulse parameters:
	dt_tsar=2e-12;   #Time Step
	T_tsar = 10e-9
	#dt_tsar = T_tsar / len(freq) # 1e-11 for 1000, 2e-12 for 5000;
	t_tsar=np.arange(0,T_tsar,dt_tsar)   #Time Vector
	t = t_tsar

	N = len(t) # 5000
	dt = t[1] - t[0]
	df = freq[1] - freq[0]
	fmin = min(freq)

	M = len(freq)
	W = np.exp(-1j*2*np.pi*dt*df);

	win = signal.tukey(M,alpha=alpha)

	fsig = win*s11
	bttczt = np.conj(chirpz(np.conj(2*fsig),1,W,N))*df
	
	phaseshift = exp(1j*2*np.pi*fmin*np.arange(0,N)*dt)

	complexsig = phaseshift*bttczt
	signal_t = np.absolute(complexsig);
	
	return signal_t
	

def chirpz(x,A,W,M):
	# doesn't seem to work when M != len(x)...

	"""Compute the chirp z-transform.
	The discrete z-transform,
	:math:`X(z) = \sum_{n=0}^{N-1} x_n z^{-n}`
	is calculated at M points,
	:math:`z_k = AW^-k, k = 0,1,...,M-1`
	for A and W complex, which gives
	:math:`X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}`
	"""
	A = np.complex(A)
	W = np.complex(W)
	if np.issubdtype(np.complex,x.dtype) or np.issubdtype(np.float,x.dtype):
		dtype = x.dtype
	else:
		dtype = float

	x = np.asarray(x,dtype=np.complex)
	
	N = x.size

	L = int(2**np.ceil(np.log2(M+N-1)))

	n = np.arange(N,dtype=float)
	y = np.power(A,-n) * np.power(W,n**2 / 2.) * x 
	Y = np.fft.fft(y,L)

	v = np.zeros(L,dtype=np.complex)
	v[:M] = np.power(W,-n[:M]**2/2.) # This is where the problem is....
	v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
	V = np.fft.fft(v)
	
	g = np.fft.ifft(V*Y)[:M]
	k = np.arange(M)
	g *= np.power(W,k**2 / 2.)

	return g


def TSAR_handle(t_tsar):
	tpulse = (t_tsar-to)*np.exp(-np.square(t_tsar-to) /np.square(tao))
	return tpulse

def dtft(x, omegas):
	"""
	Exact evaluation the DTFT at the indicated points omega for the signal x
	Note this is incredibly slow
	
	Note x runs from 0 to N-1
	"""
	N = len(x)
	ns = np.arange(N)
	W = np.zeros((len(omegas), N), dtype=np.complex128)
	for wi, w in enumerate(omegas):
		W[wi, :] = np.exp(-1.0j * w * ns)
		
	return np.dot(W, x)

#@numba.jit(nopython=True)
def nextpow2(n):
	"""
	Return the smallest power of two greater than or equal to n.
	"""
	return int(math.ceil(math.log(n)/math.log(2)))

# now try ourselves a chirp-z transform
#@numba.jit
def chirpz2(x, A, W, M):
	"""
	chirp z transform per Rabiner derivation pp1256
	x is our (complex) signal of length N
	
	
	"""
	N = len(x)
	L = 2**(nextpow2(N + M -1))  # or nearest power of two
	yn = np.zeros(L, dtype=np.complex128)
	for n in range(N):
		yn_scale =  A**(-n) * W**((n**2.0)/2.0)
		yn[n] = x[n] * yn_scale
	Yr = np.fft.fft(yn)
	
	vn = np.zeros(L, dtype=np.complex128)
	for n in range(M):
		vn[n] = W**((-n**2.0)/2.0)
		
	for n in range(L-N+1, L):
		vn[n] = W**(-((L-n)**2.0)/2.0)
		
	Vr = np.fft.fft(vn)
	
	Gr = Yr * Vr
	
	gk = np.fft.ifft(Gr)
	#gk = np.convolve(yn, vn)
	
	Xk = np.zeros(M, dtype=np.complex128)
	for k in range(M):
		g_scale = W**((k**2.0)/2.0) 
		Xk[k] = g_scale * gk[k]
		
	return Xk



def chirpz3(x, a=1.0, w=None, m=None):
	"""
	To evaluate the frequency response for the range f1 to f2 in a signal
	with sampling frequency Fs, use the following:
	m = 32;                          ## number of points desired
	w = exp(-2i*pi*(f2-f1)/(m*Fs));  ## freq. step of f2-f1/m
	a = exp(2i*pi*f1/Fs);            ## starting at frequency f1
	y = czt(x, m, w, a);
	"""
	# Convenience declarations
	ifft = FFT.ifft
	fft = FFT.fft

	if m is None:
		m = len(x)
	if w is None:
		w = np.exp(2j*N.pi/m)

	n = len(x)

	k = np.arange(m, dtype=np.float_)
	Nk = np.arange(-(n-1), m-1, dtype=np.float_)

	nfft = next2pow(min(m,n) + len(Nk) -1)
	Wk2 = w**(-(Nk**2)/2)
	AWk2 = a**(-k) * w**((k**2)/2)

	y = ifft(fft(Wk2,nfft) * fft(x*AWk2, nfft));
	y = w**((k**2)/2) * y[int(n):int(m+n)]
	return y

def next2pow(x):
	return 2**int(np.ceil(np.log(float(x))/np.log(2.0)))

