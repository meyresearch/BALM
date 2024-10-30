from tdc.multi_pred import DTI

from balm.datasets import (
    BindingDBDataset,
    LeakyPDBDataset,
    MproDataset,
    USP7Dataset,
)


DATASET_MAPPING = {
    "LeakyPDB": LeakyPDBDataset,
    "BindingDB_filtered": BindingDBDataset,
    "Mpro": MproDataset,
    "USP7": USP7Dataset,
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
        dataset = DATASET_MAPPING[dataset_name](*args, **kwargs)

    return dataset