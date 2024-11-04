# CS224W
CS224W project
### Cell line: HepG2
## Datasets used: ENCODE
[https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=intact+Hi-C&assay_title=in+situ+Hi-C&assay_title=dilution+Hi-C&assay_title=ATAC-seq&assay_title=snATAC-seq&assay_title=total+RNA-seq](https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=intact+Hi-C&assay_title=in+situ+Hi-C&assay_title=dilution+Hi-C&assay_title=ATAC-seq&assay_title=snATAC-seq&assay_title=total+RNA-seq&biosample_ontology.term_name=HepG2)
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
  - bulk chromatin accessibility

### Human reference genome GRCh38/hg38

## Commands for downloading ENCODE data
### sn-ATACseq
wget https://www.encodeproject.org/files/ENCFF724APB/@@download/ENCFF724APB.tar.gz
### Transcript levels (total RNAseq)
wget https://www.encodeproject.org/files/ENCFF103FSL/@@download/ENCFF103FSL.tsv
### 3D genome architecture (In Situ HiC data)
wget https://www.encodeproject.org/files/ENCFF913RLG/@@download/ENCFF913RLG.bigWig; \
wget https://www.encodeproject.org/files/ENCFF050EKS/@@download/ENCFF050EKS.bedpe.gz; \
wget https://www.encodeproject.org/files/ENCFF018XKF/@@download/ENCFF018XKF.bedpe.gz; \
wget https://www.encodeproject.org/files/ENCFF306VTV/@@download/ENCFF306VTV.hic
