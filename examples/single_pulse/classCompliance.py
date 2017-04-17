# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:12:43 2015

@author: fredrikeikelandfossan
"""
from __future__ import division
import numpy as np


class Laplace:
    
    def __init__(self,E,h,As,Pext,Pd):
        
        self.As=As
        self.bethaLaplace= 4*np.sqrt(np.pi)*E*h/(As*3)
        self.externalPressure=Pext
        self.Pd=Pd
    
    def A(self, P):
		P = P-self.externalPressure-self.Pd
		return (P/self.bethaLaplace + np.sqrt(self.As))** 2.
    def C(self, P):
		P = P-self.externalPressure-self.Pd
		return (2.*(P / self.bethaLaplace + np.sqrt(self.As))) / self.bethaLaplace