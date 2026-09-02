# OneMaize variant sampler audit

Status: **IMPLEMENTED / WAITING FOR REAL DATA**

Implemented behavior:

- uniform genotype sampling before class sampling;
- configurable probabilities with sum-to-one validation;
- formal fail-fast and explicit pilot renormalization for missing classes;
- SNP/small-indel event-centered crops;
- span-aware crops when an event fits in 16K;
- left/right breakpoint crops for events longer than 16K;
- bounded jitter that cannot move the target out of the model input;
- split and coordinate-genotype leakage gates;
- deterministic validation/test sampling.

No NAM26 counts or biological performance are reported because the real event files are absent.
