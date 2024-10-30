# CS224W
CS224W project
### Cell line: HepG2
## Datasets used:
ENCODE https://www.encodeproject.org/search/?type=Experiment&control_type%21=%2A&status=released&perturbed=false
- total RNAseq
- In Situ Hi-C
  - Process: Cross-links chromatin within intact nuclei, fixes the 3D structure, and then digests with restriction enzymes and performs ligation within the nucleus.
  - Advantages: Captures higher resolution contacts because the chromatin structure remains mostly undisturbed during ligation. It is the most common Hi-C method due to its robustness and reproducibility.
  - Applications: Widely used for capturing precise chromatin interactions within the nucleus, making it ideal for studying fine-scale structures like chromatin loops, TADs, and higher-order interactions.
- Intact Hi-C
  - Process: Similar to in situ Hi-C, but with even less disturbance to the native chromatin structure. The nuclei are minimally manipulated, typically using more gentle cross-linking and enzymatic treatments to preserve large structural features.
  - Advantages: Better suited for large-scale chromatin structures, potentially capturing long-range interactions with more accuracy since it preserves more of the original nuclear architecture.
  - Applications: Useful for studying larger-scale structural domains and more complex genome architecture without as much resolution at smaller interaction scales as in situ Hi-C.
- ATAC-seq
