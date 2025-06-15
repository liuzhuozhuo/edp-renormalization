# Bachelor thesis: Computational tools for perturbative calculations in renormalization of Hamiltonian. 

### Abstract
The renormalization group procedure for effective particles plays
a fundamental role in understanding physical phenomena at different scales, constructing effective theories that capture the most
relevant interactions within a certain energy range. Effective
Hamiltonians are constructed from an initial Hamiltonian and a
scale-dependent unitary transformation.
Often, a perturbative treatment is required to solve the
renormalization-group equations, and it’s useful and convenient representing the different terms of the theory as diagrams
that describe the process, in analogy with Feynman diagrams.
However, the number of diagrams increases exponentially with
the order of the perturbation, making the calculation process
tedious and error-prone.
The aim of this work is to develop a program that automates
the process of obtaining the diagrams associated with a given
interaction term, and at a given order, starting from the set of
base diagrams, given by the canonical Hamiltonian of the theory.
The program is able to discard the diagrams that do not contribute to the desired interaction, detect loops and add counterterms to the divergent contributions. By analyzing the diagrams
obtained from lower orders, and comparing with known results,
the correctness of the program has been verified. This allows to
obtain the diagrams of higher orders, and study the behavior of
the theory at those orders, where distinct features of non-Abelian
gauge theories emerge.

## Use of AI Tools

This project uses [ChatGPT](https://openai.com/chatgpt) (GPT-4o and GPT o4-mini-high) to assist in the following aspects:
- Function generation in the code (all functions from Chatgpt are indicated)
- Debugging and optimization

All AI-generated content was reviewed and verified before inclusion.

### Code description.

The basic guide of the usage of the codes in this repository can be found in `python\stepbystep-guide.ipynb`. For now the main focus is in reproducing the diagrams considering only gluons, with the functions defined in `python\functions\gluon_functions.py`.

The other file `python\functions\functions.py` is right now non-functional, the idea is for it to be valid generally, for any QFT.