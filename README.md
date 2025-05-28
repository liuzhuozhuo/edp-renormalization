# Bachelor thesis:  

### Abstract
The renormalization procedure is fundamental to understand the behavior of the 
theories at different scales, defining effective theories that describe the
behaviour of the systems for different energy levels. However, the calculations are
non-trivial, and perturbation theory is used to obtain the results, representing
the different terms of the theory as diagrams that describe the process, similar
to the Feynman diagrams. But the number of diagrams increases exponentially with
the order of the perturbation, making the calculation process tedious and prone to
errors. 

The aim of this work is to develop a program that automates the process of
obtaining the diagrams associated with a given process, up to a given order,
starting from a set of base diagrams, called canonical diagrams given by the
theory. The program is able to discard the diagrams that do not contribute to the
process, detect loops and add counterterms to the diagrams to cancel divergences. 

By analysing the diagrams obtained from lower orders, and comparing with known
results, the correctness of the program has been verified. This allows to obtain
the diagrams of higher orders, and study the behavior of the theory at those
orders, where unique phenomena of non-abelian theories can be observed.

### Code description.

The basic guide of the usage of the codes in this repository can be found in `python\stepbystep-guide.ipynb`. For now the main focus is in reproducing the diagrams considering only gluons, with the functions defined in `python\functions\gluon_functions.py`.

The other file `python\functions\functions.py` is right now non-functional, the idea is for it to be valid generally, for any QFT.