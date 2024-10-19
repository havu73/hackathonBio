# hackathonBio

# PhageBook - BioML Hackathon Project

**Team Name:** PhageBook

**Project Duration:** 10 days

**Goal:** Utilizing Evo to predict phage-bacteria host interactions for antibiotic design.

# Table of Contents
1. Introduction 
2. Problem Statement   
3. Approach
4. Data Collection and Processing
5. Modeling and Prediction
6. Results and Evaluation
7. Challenges Faced
8. Conclusion
9. Future Work
10. Team Members

# 1. Introduction
PhageBook is an innovative platform designed to predict phage-bacteria host interactions using evolutionary models. By leveraging the Evo framework, we aim to design phage-based solutions that combat antibiotic-resistant bacteria.

# 2. Problem Statement
The increasing prevalence of antibiotic-resistant bacteria is a global health concern. Current antibiotics are becoming less effective, and there is an urgent need for novel therapeutic solutions. Phage therapy offers a promising alternative by using viruses that specifically target and kill bacteria. However, identifying the right phage for a given bacterial strain remains challenging. PhageBook aims to address this by predicting phage-bacteria host interactions.

# 3. Approach

# 4. Data Collection and Processing
We used several publicly available datasets for phage-host pairs. Key steps in data collection and processing include: 
Phage and Bacterial Data: We collected four experimentally validated datasets (PhagesDB, Klebsiella, Vibrio, and E. coli) and one with predicted pairs (PhageScope). 

### 4.1 Curation: 
* For experimental studies of individual species, their in vitro verified interactions were collected from the supplementary tables of the manuscripts ([Klebsiella](https://doi.org/10.1038/s41467-024-48675-6), [Vibrio](https://doi.org/10.1038/s41467-021-27583-z), and [E. coli](https://doi.org/10.1101/2023.11.22.567924)). Phage genome sequence data and bacterial genome sequence data were retrieved from RefSeq and Genbank according to the unique IDs provided by the study authors. In the end, for these datasets, we have pairs between:

 * **Vibrio: 259 bacterial hosts and 239 phages**

 * **Klebsiella: 213 bacterial hosts and 120 phages**

 * **E.coli: 325 bacterial hosts and 96 phages**

* For PhagesDB, we retrieved phage-host pairs and phage genome sequences from their [web platform](https://phagesdb.org/) and separately retrieved bacterial genome sequences from RefSeq using a taxonomy search. Ultimately, we have interactions between **XXX unique phages and XXX bacterial species.**
* For PhageScope, we retrieved phage-host pairs and phage genome sequences from their [web platform](https://phagescope.deepomics.org/database). We filtered phages to the subset of these with complete genome sequences and lylic lifestyle. The bacterial host data specified only the host taxonomy name, so we filtered the data to include only hosts with full species-level taxonomy known (the most specific level of host information available in this database). We also separately retrieved bacterial genome sequences from RefSeq using a taxonomy search and picked the genome sequence of the best quality if multiple genomes were available. Ultimately, we have interactions between **4434 unique phages and 180 bacterial host species.** 

# 5. Modeling and Prediction 
### 5.1. Feature Extraction
### 5.2. Modeling

# 6. Results and Evaluation
# 7. Challenges Faced

# 8. Conclusions

# 9. Future Work

# 10. Team Members

**Ha Vu:** Postdoctoral Scholar, Gladstone Institutes
* **Team lead,** pipeline construction, feature extraction

**Veronika Dubinkina:** Bioinformatics fellow, Gladstone Institutes
* Data curation and domain knowledge

**Boyang Fu:** PhD Candidate, UCLA
* Model training
  
**Emily Maciejewski:** PhD Candidate, UCLA
* Model training
  
**Khoa Hoang:** PhD Student, Stanford
* Model training
  
**Tung Nguyen:** PhD Candidate, UCLA
* Feature extraction, model training
  
**Cindy K. Pino:** PhD Candidate, UCSF/UCB/Gladstone Institutes
* Data curation and domain knowledge




