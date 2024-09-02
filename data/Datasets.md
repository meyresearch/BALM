# Datasets Overview

Each of the following CSV files contains three key columns: `Sequence`, `SMILES`, and `Binding Affinity score (pKD)`. These columns represent the protein sequence, ligand's SMILES notation, and the experimentally determined binding affinity, respectively. Below is a brief description of each dataset along with the number of interactions they contain.

1. **BindingDB_filtered.csv**
   - **Description**: A curated dataset derived from the BindingDB database, focusing on high-quality, experimentally validated protein-ligand interactions.
   - **Number of Interactions**: 24,700

2. **CATS.csv**
   - **Description**: This dataset includes data relevant to the CATS descriptor, which is used for capturing chemical and topological properties of small molecules in relation to their binding affinity with proteins.
   - **Number of Interactions**: 136

3. **HSP9.csv**
   - **Description**: A focused dataset on Heat Shock Protein 90 (HSP90), containing binding affinity information specific to this protein and its associated ligands.
   - **Number of Interactions**: 180

4. **leaky_pdb.csv**
   - **Description**: Derived from the PDBBind database, this dataset is curated to focus on high-confidence protein-ligand complexes, minimizing noise and emphasizing reliable interactions.
   - **Number of Interactions**: 19,443

5. **Mpro.csv**
   - **Description**: This dataset is specific to the SARS-CoV-2 main protease (Mpro), encompassing binding affinity data for various small molecules tested against this critical COVID-19 drug target.
   - **Number of Interactions**: 2,749

6. **USP7.csv**
   - **Description**: A dataset focused on Ubiquitin-Specific Protease 7 (USP7), an important target in cancer therapy, containing binding affinity data for ligands interacting with USP7.
   - **Number of Interactions**: 1,799
