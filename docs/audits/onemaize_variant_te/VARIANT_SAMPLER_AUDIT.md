# OneMaize variant sampler audit

Status: **EXPERIMENTAL / IMPLEMENTED / WAITING FOR REAL DATA**

This sampler is not required by the formal schema-v3 all-cultivar Model B.

Implemented behavior:

- configured class sampling followed by uniform genotype sampling within the
  genotypes that genuinely contain that class;
- configurable probabilities with sum-to-one validation;
- global class fail-fast and explicit pilot renormalization when an entire class
  is unavailable; B73 is not required to have B73-vs-B73 variant events;
- SNP/small-indel event-centered crops;
- span-aware crops when an event fits in 16K;
- left/right breakpoint crops for events longer than 16K;
- bounded jitter that cannot move the target out of the model input;
- split and coordinate-genotype leakage gates;
- deterministic validation/test sampling.

No NAM26 counts or biological performance are reported because the real event files are absent.
