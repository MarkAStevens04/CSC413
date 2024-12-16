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
  - For data pre-processing, leveraged saving to different files to allow one large pre-processing run to be completed, and never performed again. Significantly improves speed of algorithm. Only saved relevant information to improve maximum capacity. Can specify set of protein codes & use it to train or evaluate models.
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
1. Clone the repository onto your device
2. Downloaded required packages with `pip install -r requirements.txt`


3. Install pytorch with cuda. Documentation for this installation can be found [here](https://pytorch.org/get-started/locally/). 
Helpful debugging documentation can be found [here](https://saturncloud.io/blog/pytorch-says-that-cuda-is-not-available-troubleshooting-guide-for-data-scientists/#:~:text=Check%20your%20PyTorch%20installation%3A%20If,ensure%20that%20it's%20installed%20correctly.&text=This%20will%20list%20any%20CUDA%2Drelated%20errors%20in%20your%20system%20logs).
 
   Please Note: Some packages may not be listed in requirements.txt and will require manual download.
This code was run with Python 3.11.

# How to use
### Data Processing
Run `parse_seq.py` to automatically download and parse files. Note that you may need to run `transformer.py` for automatic creation of directories.

Protein IDs can be specified in `PDBs/protein-ids.txt`. Protein ids are 4-letter codes for downloading the protein's structure from the RCSB.

Pre-processed data will automatically be saved to `PDBs/pre_processed_data`. If downloading many proteins (>1,000) it's advisable to run this as soon as possible, as the data pre-processing may take a while.

### Training
To train a model, run `transformer.py`. If running from the terminal, you may optionally add arguments.

`transformer.py <Node name> <reprocess> <data size> <model heads> <model depth>`

- **Node name**: Specify the name that this model is training on. Useful when concurrently training multiple models.
- **Reprocess**: Either 'y' or 'f'. If 'y', will run `parse_seq.py` to re-download and pre-process all data. If 'f', will skip this step.
- **Data size**: Number of proteins from your pre-processed data you would like to include. Allows for debugging with small sample size without augmenting previously performed data pre-processing.
- **Model heads**: Number of heads in our multi-head transformer architecture.
- **Model depth**: Number of attention heads we run successively.
