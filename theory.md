# Theoretical framework

## Renormalization group procedure for effective particles (RGPEP)

The RGPEP is a renormalization procedure applied within the Hamiltonian formulation of quantum field theories. By considering a series of unitary transformations applied to the canonical Hamiltonian

In RGPEP a parameter $s$ is considered to be the effective size of the particles considered, so $s=0$ 
This way effective Hamiltonian $\mathcal{H}_s$ can be defined.

Due to dimensional and notational reasons, it is convenient to consider the scale parameter $t = s^4$ instead

The effective particles are defined by the effective particle operators, these differ from the canonical ones by the unitary transformation $\mathcal{U}_t$

$$
a_t = \mathcal{U}_t a_0\mathcal{U}_t^\dagger
$$

The effective Hamiltonian, written in terms of the effective operators, is related to the canonical Hamiltonian plus counterterms by the condition,

$$
\mathcal{H}_0 (a_0) =\mathcal{H}_s(a_s)
$$

Expressing the effective Hamiltonian in terms of the canonical particle operators,

$$
\mathcal{H}_t(a_0) = \mathcal{U}_s^\dagger \mathcal{H}_0(a_0) \mathcal{U}_s
$$

Differentiating with respect to t, one obtain the RGPEP equation.

$$
\mathcal{H}_t^\prime (a_0) = [\mathcal{G}_t(a_0), \mathcal{H}_t(a_0)],
$$
where $\mathcal{G}_t$ is the generator of the transformation.






