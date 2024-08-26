# BALM: Binding Affinity Predictions with Protein and Ligand Language Models

## Overview

Reliable in-silico methods for predicting protein-ligand binding affinity are pivotal for accelerating the hit identification from vast molecular libraries. While structure-based methods like giga-docking are commonly used, they are unreliable for rank-ordering hit compounds, and free energy-based methods, though accurate, are computationally expensive. Existing deep learning models struggle to generalize to unseen targets or drugs. We introduce **BALM**, a sequence-based deep learning framework for predicting **b**inding **a**ffinity using pretrained protein and ligand **l**anguage **m**odels. BALM learns by optimizing the distance between protein and ligand embeddings in a shared space using the cosine similarity metric that directly represents experimental binding affinity. We incorporate parameter-efficient fine-tuning methods in the BALM framework to adapt pretrained protein and ligand language models for binding prediction. Our findings on the BindingDB and Leak Proof PDBBind dataset demonstrate BALM's potential for generalizing to unseen drugs or targets.

### To-do's
1. Setup HF BALM benchmark, documentation should specify description for columns.
2. Few shot training notebook (colab) using pretrained Binding DB model (we can gove them option to select the model). All the models will be in HF models (bindingdb clean, original binding db, and leakypdb)
3. Zero shot notebook, we will have to define the csv input format for the user.
4. New dataset traininig script (bash script), also a script to train BindingDB model again.
5. Check the environment installation and will give a docker image.

