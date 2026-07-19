import datasets

hf_dataset_id = "ucrelnlp/USAS-WSD"

print("Dataset config names:")
print(datasets.get_dataset_config_names(hf_dataset_id))
print()

print("Dataset splits:")
print(datasets.get_dataset_split_names(hf_dataset_id, "eng"))
print()

print("Dataset features")
print(datasets.load_dataset_builder(hf_dataset_id, "eng").info.features)

dataset = datasets.load_dataset(hf_dataset_id, "zho", split="test")
print(dataset)
print(dataset[0]['text'])
#dataset = load_dataset("ucrelnlp/USAS-WSD",, split="train")

print(dataset.builder())