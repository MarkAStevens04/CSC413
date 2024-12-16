![Alt text](logo.png?raw=true "Sigma Fold with 4 computer programmers")

A transformer architecture, from start to finish.

# Overview of Files
- **get_pdbs.py** : Download mmCif files, extract atom positions & protein sequence.
- **amino_expert.py** : Utilize "standard aminos" to assist with amino-atom-index interconversions.
- **parse_seq.py** : Align full protein sequence with known atom sequence. Data pre-processing.
- **transformer.py** : Final data processing & Transformer training. Code for concurrent trainings & CUDA acceleration.
- **eval_model.py** : Take model save-states & obtain training & validation accuracies at different time points.

# Highlights
- Atom-Level Intrinsically Disordered Region Identification: 
  - Used mmCIF parsing to extract atom positions. Compared this against known Amino Acid sequence and "Ideal Aminos". From this, determined which atoms were "supposed" to be included, but were not, and classified these as Intrinsically Disordered.
- CUDA Acceleration with Pytorch
  - Utilized pytorch with CUDA to enable massive parallelization with GPUs. Significantly improves training & inference time.
- Memory Mapping for Large Datasets
  - Utilized memory mapping to allow large datasets to be stored on hard drive and be taken to correct device only when used in that batch. Allows for datasets which far surpass batch size, without holding entire dataset in memory and without constantly opening and closing files.
- Multi-Node Management
  - We manage concurrent training of multiple models, even if worker nodes utilize a Network File System (NFS). Allows for synchronous model training, speeding up hyper-parameter search. 
- Big-Data Management
  - For data pre-processing, leveraged saving to different files to allow one large pre-processing run to be completed, and never performed again. Significantly improves speed of algorithm. Only saved relevant information to improve maximum capacity.
- Extensive Logging
  - To enable quick debugging in multi-node environments, we use extensive data logging to track errors as they occur. We 'fail gracefully', allowing for this large project to handle unexpected cases & continue training even if exceptions occur.
- Model Saves
  - Save models during training with configurable save frequencies. Allows for model to be recovered in case of node failure. Also enables retroactive performance analysis by iterating through every saved model & testing performance at that stage of training.
- ESM Embedding Integration
  - Utilizes existing architecture (ESM Fold) for amino acid embedding. Accelerates training by reducing time of model warm-up.
- Extensive Model Analysis
  - Utilizes vast array of established model analysis tools, including RMSD, TM-Score and GDT.
- Protein Rendering
  - Framework established for rendering proteins from their predicted positions. Uses existing mmCif file & modifies positions of each atom.

# How to install
Clone the repository, then downloaded required packages with

'pip install -r requirements.txt'

# How to use