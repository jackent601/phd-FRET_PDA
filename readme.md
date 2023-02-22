Python implementation of Probability distribution analysis

#### Some References

- [Characterizing Single-Molecule FRET Dynamics with Probability Distribution Analysis][4],
  - discusses adding gaussian noise, includes another reference
- [Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States][3]
  - [ref] (Master Paper?) Separating Structural Heterogeneities from Stochastic Variations in Fluorescence Resonance Energy Transfer Distributions via Photon Distribution Analysis
- [Conformational Dynamics of DNA Hairpins at Millisecond Resolution Obtained from Analysis of Single-Molecule FRET Histograms][2], provide a non-derived equation for shot-noise dependent burst

#### PDA Algorithm

Taken from [supplementary information][1] to [Structural Dynamics of DNA Hairpin ][2]

1. Choose an oversampling factor N, a physical model that describes the dynamics of the molecule and realistic initial parameters (in our case: E<sub>op</sub>, E<sub>cl</sub>, k<sub>op</sub> and k<sub>cl</sub>).
2. For each burst, calculate the overall time the molecule spent in each of the two states (τ<sub>op</sub> and τ<sub>cl</sub>) using a Monte Carlo simulation (running for a time duration equal to the burst duration, BD, of the corresponding burst) (**See Rational**). Here Kinetic model taken from [Detection of Structural Dynamics by FRET paper][3]
3. Calculate the shot-noise-free E value (Esnf) using Eq. 1.

$$
E_{snf} = \frac{\tau_{op}\cdot E_{op} + \tau_{cl}\cdot E_{cl}}{BD}
$$

4. Calculate the shot-noise-dependent E value (Esn) using Eq. 3.


5. Repeat steps (ii) through (iv) N times for each burst and add the results to an E-histogram.
6. Divide the resultant E-histogram by N.
7. Improve on E<sub>op</sub>, E<sub>cl</sub>, k<sub>op</sub>, and k<sub>cl</sub> to achieve best agreement between the calculated and the experimental E-histograms using a chi-squared minimization. The histograms may be slightly smoothed to assist in the optimization

    7.a. [Characterizing Single-Molecule FRET][4] section 2.5 provides a good explanation of whether a good fit is found (sometimes problems with degeneracy in parameter surface)

[1]: https://pubs.acs.org/doi/suppl/10.1021/jp411280n/suppl_file/jp411280n_si_001.pdf "SI - Conformational Dynamics of DNA Hairpins at Millisecond Resolution Obtained from Analysis of Single-Molecule FRET Histograms"

[2]: https://pubs.acs.org/doi/10.1021/jp411280n "Conformational Dynamics of DNA Hairpins at Millisecond Resolution Obtained from Analysis of Single-Molecule FRET Histograms"

[3]: https://pubs.acs.org/doi/10.1021/jp102156t "Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States"

[4]: https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cphc.201000129 "Characterizing Single-Molecule FRET Dynamics with Probability Distribution Analysis"



#### ToDo

- [ ] Single burst analysis
- [ ] Wrapper for gradient descent
- [ ] Gaussian Noise
- [ ] Some functions now depreciated in PDA.py file and should be removed
- [ ] Specify a config to clean up some functions

#### 'Rational'

**(2)** [...] the Monte Carlo simulation (step (ii)) was performed according to a two-state model. **The simulation stochastically draws open and closed states dwell-times from exponentially distributed dwell-times (with typical k<sub>cl</sub> and k<sub>op</sub> rates, respectively) for overall time duration equal to the burst duration**. The overall times the molecule spent in each state (τ<sub>op</sub> and τ<sub>cl</sub>) were then calculated by summing over all the open and the closed state dwell-times. Notice that τ<sub>op</sub> and τ<sub>cl</sub> are the sum over all the durations a simulated molecule spends in the open and closed states, not the typical opening and closing times (i.e., not 1/k<sub>op</sub> or 1/k<sub>cl</sub>).

#### PDA Other Notes:

**[Sensitivity]**[4] We note that the PDA method ismost sensitive when 0.01 << (tr/TD) << 10, whereTDis the averagedwell time in the confocal volume (see the Supporting Infor-mation, Figure S-3). To achieve the highest sensitivity using the PDA method, one can tune the value ofTDto match the rangeof the conformational rates to be explored. This can be accom-plished by increasing the size of the confocal spot, by usingpolyacrylamide gels, or by changing the viscosity of the sur-rounding medium.

**[Gaussian Broadening]**[2],[4][4] Experiments to date have much larger width E w.r.t. shot-noise theory. Typically E values are broadened by sampling from a gaussian distribution with $\sigma_{i}$ in the case of [2][2] 1.4 angstroms which best fit results (need to use R_{0} values to calcualte)