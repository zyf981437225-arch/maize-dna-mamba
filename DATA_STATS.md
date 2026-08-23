# Maize DNA data status

No maize/B73 FASTA is present in the local workspace, so real maize base and
window counts are not available yet. Formal DNA pretraining is therefore not
ready and has not been started.

`scripts/prepare_maize_genome.py` writes the authoritative, data-derived report
to `<prepared-output>/DATA_STATS.md`. That generated report includes:

- genome names, total bp, and contig count;
- A/C/G/T/N percentages;
- raw U percentage;
- other IUPAC ambiguity and invalid-character percentages;
- 10,240-bp window size and configured stride;
- retained and filtered window counts for train/validation/test;
- whole-contig split counts and formal-training readiness gates.

The dataset loader refuses training if the raw U-frequency gate fails. The
formal data module also requires non-empty train, validation, and test splits.
This file should be replaced or supplemented with the generated server report
after the B73 pilot FASTA is prepared.
