# Some math functions which aren't as easily accessible through numpy
	# David Garrett - October 2017

import numpy as np
import matplotlib.pyplot as plt

# TSAR pulse parameters:
dt_tsar=2e-12;   #Time Step
T_tsar= 10e-9;     #Time Length
t_tsar=np.arange(0,T_tsar,dt_tsar)   #Time Vector

# Pulse Parameters
tao=62.5e-12;
to=4*tao;


def inverseczt_charlotte(freq,s11,t=t_tsar):
	N = len(t)
	dt = t[1] - t[0]
	df = freq[1] - freq[0]
	fmin = min(freq)
	M = len(freq)
	W = np.exp(-1j*2*np.pi*dt*df)
	A = np.exp(1j*2*np.pi*fmin*dt)

	tpulse = TSAR_handle(t)
	fpulse = chirpz(tpulse, A,W,M)
	fpulse = fpulse/max(np.absolute(fpulse))
	fsig = fpulse*s11

	A = np.exp(1j*2*np.pi*t[0]*df)
	bttczt = np.conj(chirpz(np.conj(fsig),A,W,N))*df*np.sqrt(2)
	phaseshift = np.exp(1j*2*np.pi*fmin*np.arange(0,N)*dt)
	complexsig = phaseshift*bttczt
	tsig = np.real(complexsig)
	
	return t, tsig



def chirpz(x,A,W,M):
    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}    
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
    v[:M] = np.power(W,-n[:M]**2/2.)
    v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)
    
    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W,k**2 / 2.)

    return g


def TSAR_handle(t_tsar):
    tpulse = (t_tsar-to)*np.exp(-np.square(t_tsar-to) /np.square(tao))
    return tpulse
