# References

The formulations implemented in OpenSG-TW are taken from the literature below; the theory pages cite these
by name. No formulation in this documentation is original or speculative — each is traceable to one of
these sources (DOIs are given where verified).

## Mechanics of Structure Genome, VAM & beam sectional analysis

- W. Yu. *Mechanics of Structure Genome (MSG)*. Purdue University. (The structure-genome dimensional-reduction
  framework underlying all three OpenSG-TW models.)
- W. Yu, D. H. Hodges, J. C. Ho. "Variational asymptotic beam sectional analysis — an updated version."
  *International Journal of Engineering Science* **59** (2012) 40–64. DOI [10.1016/j.ijengsci.2012.03.006](https://doi.org/10.1016/j.ijengsci.2012.03.006).
  *(The VABS Euler–Bernoulli 4×4 / Timoshenko 6×6 used as the solid reference.)*
- W. Yu, V. V. Volovoi, D. H. Hodges, X. Hong. "Validation of the variational asymptotic beam sectional
  analysis." *AIAA Journal* **40**(10) (2002) 2105–2112. DOI [10.2514/2.1545](https://doi.org/10.2514/2.1545).
- A. Deo, W. Yu. Thin-walled cross-sectional analysis via the Mechanics of Structure Genome.
  *(The MSG-TW shell blueprint: zeroth-order = in-plane plate ABD, section-shear at $\mathcal{O}(\zeta^2)$.)*
- A. Bagla *et al.* Multi-cell thin-walled composite beam homogenization (ASC). *(The two-cell $[-45]$
  benchmark and the section-shear energy ordering used by the RM model.)*

## Reissner–Mindlin shell, MITC & assumed natural strain

- E. N. Dvorkin, K.-J. Bathe. "A continuum mechanics based four-node shell element for general non-linear
  analysis." *Engineering Computations* **1**(1) (1984) 77–88. DOI [10.1108/eb023562](https://doi.org/10.1108/eb023562). *(Origin of MITC.)*
- K.-J. Bathe, E. N. Dvorkin. "A four-node plate bending element based on Mindlin/Reissner plate theory and
  a mixed interpolation." *IJNME* **21**(2) (1985) 367–383. DOI [10.1002/nme.1620210213](https://doi.org/10.1002/nme.1620210213). *(MITC4.)*
- K.-J. Bathe, E. N. Dvorkin. "A formulation of general shell elements — the use of mixed interpolation of
  tensorial components." *IJNME* **22**(3) (1986) 697–722. DOI [10.1002/nme.1620220312](https://doi.org/10.1002/nme.1620220312). *(Names MITC.)*
- M. L. Bucalem, K.-J. Bathe. "Higher-order MITC general shell elements." *IJNME* **36**(21) (1993)
  3729–3754. DOI [10.1002/nme.1620362109](https://doi.org/10.1002/nme.1620362109).
- P.-S. Lee, K.-J. Bathe. "The quadratic MITC plate and MITC shell elements in plate bending."
  *Advances in Engineering Software* **41**(5) (2010) 712–728. DOI [10.1016/j.advengsoft.2009.12.011](https://doi.org/10.1016/j.advengsoft.2009.12.011).
- J. C. Simo, T. J. R. Hughes. "On the variational foundations of assumed strain methods."
  *Journal of Applied Mechanics* **53**(1) (1986) 51–54. DOI [10.1115/1.3171737](https://doi.org/10.1115/1.3171737). *(Hu–Washizu basis of assumed strain.)*
- E. Carrera, A. G. de Miguel, A. Pagani. "Extension of MITC to higher-order beam models and shear locking
  analysis." *IJNME* **112**(13) (2017) 1889–1908. DOI [10.1002/nme.5588](https://doi.org/10.1002/nme.5588). *(1-D MITC beam.)*
- J. N. Reddy. "On locking-free shear deformable beam finite elements." *CMAME* **149**(1–4) (1997)
  113–132. DOI [10.1016/S0045-7825(97)00075-3](https://doi.org/10.1016/S0045-7825(97)00075-3).

## Tying points, field consistency & reduced integration

- J. Barlow. "Optimal stress locations in finite element models." *IJNME* **10**(2) (1976) 243–251.
  DOI [10.1002/nme.1620100202](https://doi.org/10.1002/nme.1620100202). *(Tying points: $\xi=0$ linear; $\xi=\pm1/\sqrt3$ quadratic.)*
- G. Prathap, G. R. Bhashyam. "Reduced integration and the shear-flexible beam element." *IJNME* **18**(2)
  (1982) 195–210. DOI [10.1002/nme.1620180205](https://doi.org/10.1002/nme.1620180205). *(p=1: reduced ≡ field-consistent.)*
- G. Prathap. "Field-consistent strain interpolations for the quadratic shear flexible beam element."
  *IJNME* **23**(11) (1986) 1973–1984. DOI [10.1002/nme.1620231102](https://doi.org/10.1002/nme.1620231102). *(p=2: reduced ≠ MITC.)*
- G. Prathap. *The Finite Element Method in Structural Mechanics*. Kluwer (1993). DOI [10.1007/978-94-017-3319-9](https://doi.org/10.1007/978-94-017-3319-9).
- T. J. R. Hughes. *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*.
  Prentice-Hall (1987). *(B-bar, selective reduced integration.)*
- O. C. Zienkiewicz, R. L. Taylor, J. M. Too. "Reduced integration technique in general analysis of plates
  and shells." *IJNME* **3**(2) (1971) 275–290. DOI [10.1002/nme.1620030211](https://doi.org/10.1002/nme.1620030211).

## Soft-core / sandwich physics (why FSDT-MITC is necessary but not sufficient)

- N. J. Pagano. "Exact solutions for rectangular bidirectional composites and sandwich plates."
  *Journal of Composite Materials* **4**(1) (1970) 20–34. DOI [10.1177/002199837000400102](https://doi.org/10.1177/002199837000400102).
- H. Altenbach, V. A. Eremeyev, K. Naumenko. "On the use of the first order shear deformation plate theory
  for the analysis of three-layer plates with thin soft core layer." *ZAMM* **95**(10) (2015) 1004–1011.
  DOI [10.1002/zamm.201500069](https://doi.org/10.1002/zamm.201500069).
- E. Carrera. "Historical review of zig-zag theories for multilayered plates and shells."
  *Applied Mechanics Reviews* **56**(3) (2003) 287–308. DOI [10.1115/1.1557614](https://doi.org/10.1115/1.1557614).
- A. Tessler, M. Di Sciuva, M. Gherlone. "A refined zigzag beam theory for composite and sandwich beams."
  *Journal of Composite Materials* **43**(9) (2009) 1051–1081. DOI [10.1177/0021998308097730](https://doi.org/10.1177/0021998308097730).

## Software

- **OpenSG** — W. Yu group, Purdue University. <https://github.com/wenbinyugroup/OpenSG>. *(The FEniCS
  reference implementation OpenSG-TW extends with a pure-JAX, thin-walled back-end.)*
- **OpenSG_io** — windIO / PreVABS / OpenFAST → OpenSG cross-section converter. <https://github.com/bagla0/OpenSG_io>.
