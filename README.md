# wlai-uf-hackathon-2023

## Overview

Machine learning code for quality control of water level data from measurement stations.

## Contact

For additional information, contact:

- Lindsay Abrams,
- NOAA Center for Operational Oceanographic Products and Services,
- lindsay.abrams@noaa.gov

## Tutorial

### Download data

1) Install `gdown` to download from Google Drive

    pip install gdown


2) Download all 5 data archives 

    # This does not work because of the share link permissions! 
    sh download_data.sh


### Setup Python environment 

1) [Install Anaconda using their documentation](https://docs.anaconda.com/free/anaconda/install/linux/)


2) Create Anaconda environment from requirements file
<!-- end of the list -->

    conda env create --file environment.yml


3) Activate Anaconda environment
<!-- end of the list -->

    conda activate wlai


4) Deactive Anaconda environment (when you are finished working)
<!-- end of the list -->

    conda deactivate


### Run model on PC workstation

1) Run simple example (single station with minimal options)

- `qc_model_nn.py` contains neural network architecture and related utilities.
- It also has a `main` function that demonstrates how to use the functions to train a model.
<!-- end of the list -->

    python qc_model_nn.py \ 
        --station 9751639 \
        --directory data/ \
        --epochs 5        \
        --batch_size 256


### Run model on Slurm-based HPC


## NOAA Open Source Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
