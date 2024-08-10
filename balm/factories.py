from tdc.multi_pred import DTI

from balm.datasets import (
    BindingDBDataset,
    CATSDataset,
    HSP9Dataset,
    LeakyPDBDataset,
    MproDataset,
    USP7Dataset,
)


DATASET_MAPPING = {
    "LeakyPDB": {
        "filepath": "data/leaky_pdb.csv",
        "class": LeakyPDBDataset,
    },
    "BindingDB_filtered": {
        "filepath": "data/BindingDB_filtered.csv",
        "class": BindingDBDataset
    },
    "Mpro": {
        "filepath": "data/Mpro.csv",
        "class": MproDataset
    },
    "USP7": {
        "filepath": "data/USP7.csv",
        "class": USP7Dataset
    },
    "HSP9": {
        "filepath": "data/HSP9.csv",
        "class": HSP9Dataset
    },
    "CATS": {
        "filepath": "data/CATS.csv",
        "class": CATSDataset
    },
}


def get_dataset(dataset_name, harmonize_affinities_mode, *args, **kwargs):
    if dataset_name.startswith("DTI_"):
        dti_dataset_name = dataset_name.replace("DTI_", "")
        dataset = DTI(name=dti_dataset_name)
        if harmonize_affinities_mode:
            dataset.harmonize_affinities(
                mode=harmonize_affinities_mode
            )
            # Convert $K_d$ to $pKd$
            dataset.convert_to_log(form="binding")
    else:
        dataset = DATASET_MAPPING[dataset_name]["class"](*args, filepath=DATASET_MAPPING[dataset_name]["filepath"], **kwargs)

    return dataset