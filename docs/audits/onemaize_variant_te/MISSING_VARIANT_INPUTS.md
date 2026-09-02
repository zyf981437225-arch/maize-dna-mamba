# Missing variant inputs for OneMaize Model B

Status: **BLOCKED BY MISSING VARIANT DATA**

The repository contains no real NAM26 VCF/BCF, SV, PAV, cultivar-specific TE insertion/deletion, TE-family polymorphism, or validated assembly-to-B73 coordinate-mapping files. Different cultivar FASTA sequences are not explicit variant annotation.

Formal Model B preparation requires, for every genotype entering a split:

1. A frozen genotype split consistent with schema-v3 (`23 train / 1 val / 2 test`, B73 in train).
2. Variant events whose coordinates index that genotype's training FASTA, or a validated transformation that produces such coordinates.
3. SNP and small-indel calls with caller/version and reference/alternate alleles.
4. SV breakpoints/intervals with explicit event type and coordinate convention.
5. PAV intervals plus a valid coordinate mapping; gene names alone are insufficient.
6. Cultivar-specific TE insertion/deletion events. The existing TE GFF3 only describes repeat occupancy and cannot establish polymorphism.
7. TE family/superfamily fields if family-specific analysis is required.
8. Matching FASTA, FAI/GZI, gene GFF3 and TE GFF3 for every genotype.

Current parser support is limited to standard VCF/VCF.GZ records. Laboratory-specific SV/PAV/TE tables need a source-specific adapter after their real headers, semantics and coordinate system are available. Do not rename columns to imitate schema-v4 without validating their meaning.
