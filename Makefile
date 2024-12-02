# Makefile to download ENCODE data files into a 'data' directory

DATA_DIR = data
TAR_FILE = $(DATA_DIR)/ENCFF724APB.tar.gz
TSV_FILE = $(DATA_DIR)/ENCFF103FSL.tsv
BIGWIG_FILE = $(DATA_DIR)/ENCFF913RLG.bigWig
BEDPE_FILE_1 = $(DATA_DIR)/ENCFF050EKS.bedpe.gz
BEDPE_FILE_2 = $(DATA_DIR)/ENCFF018XKF.bedpe.gz
PEAKS_BED = $(DATA_DIR)/ENCFF439EIO.bed.gz
ATAC_BW = $(DATA_DIR)/ENCFF262URW.bigWig
REFERENCE_GENOME = $(DATA_DIR)/hg38.fa.gz
CHROM_SIZES = $(DATA_DIR)/hg38.chrom.sizes
EMBEDDINGS = $(DATA_DIR)/embeddings.npz
EXPRESSION = $(DATA_DIR)/expression.tsv
GENE_ANNOTATION = $(DATA_DIR)/gencode.v38.chr_patch_hapl_scaff.basic.annotation.gtf.gz

# Create data directory
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Download transcript levels (total RNAseq)
$(TSV_FILE): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF103FSL/@@download/ENCFF103FSL.tsv

# Download 3D genome architecture (In Situ HiC data)
$(BIGWIG_FILE): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF913RLG/@@download/ENCFF913RLG.bigWig

$(BEDPE_FILE_1): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF050EKS/@@download/ENCFF050EKS.bedpe.gz

$(BEDPE_FILE_2): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF018XKF/@@download/ENCFF018XKF.bedpe.gz


# Download Transcript levels (total RNAseq) archive
$(TAR_FILE): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF724APB/@@download/ENCFF724APB.tar.gz

$(PEAKS_BED): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF439EIO/@@download/ENCFF439EIO.bed.gz

$(ATAC_BW): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF262URW/@@download/ENCFF262URW.bigWig

$(REFERENCE_GENOME): | $(DATA_DIR)
	wget -O $@ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

$(CHROM_SIZES): | $(DATA_DIR)
	wget -O $@ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes

$(EMBEDDINGS): | $(DATA_DIR)
	wget -O $@ https://mitra.stanford.edu/kundaje/kobbad/CS224W/embeddings.npz

$(EXPRESSION): | $(DATA_DIR)
	wget -O $@ https://www.encodeproject.org/files/ENCFF336WOX/@@download/ENCFF336WOX.tsv

$(GENE_ANNOTATION): | $(DATA_DIR)
	wget -O $@ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.chr_patch_hapl_scaff.basic.annotation.gtf.gz

# gunzip hg38.fa.gz
$(DATA_DIR)/hg38.fa: $(REFERENCE_GENOME)
	gunzip -c $< > $@

# gunzip gencode.v38.annotation.gtf.g
$(DATA_DIR)/gencode.v38.chr_patch_hapl_scaff.basic.annotation.gtf: $(GENE_ANNOTATION)
	gunzip -c $< > $@

# Main target to download all files
all: $(TAR_FILE) $(TSV_FILE) $(BIGWIG_FILE) $(BEDPE_FILE_1) $(BEDPE_FILE_2) $(PEAKS_BED) $(ATAC_BW) $(REFERENCE_GENOME) $(CHROM_SIZES) $(EMBEDDINGS) $(EXPRESSION) $(GENE_ANNOTATION) 

# Clean downloaded files
clean:
	rm -rf $(DATA_DIR)
