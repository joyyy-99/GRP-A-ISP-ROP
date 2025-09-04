from torch.utils.data import DataLoader
from src.rop_pipeline.dataset import ROPDataset

ds = ROPDataset("data/metadata/parsed.csv", train=True, label_col="dg")
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
x, y, meta = next(iter(dl))
print("images:", x.shape)       # expect [8, 3, 224, 224]
print("labels:", y.shape)       # expect [8]
print("first file:", meta["filepath"][0])
