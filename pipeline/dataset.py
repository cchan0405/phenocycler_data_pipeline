from torch.utils.data import DataLoader, Dataset

class DataCharacteristic(Dataset):
    def __init__(self, filteredtiles):
        self.data = filteredtiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def loader_creation(filtered_tiles):
    processed_tiles = DataCharacteristic(filtered_tiles)
    loader = DataLoader(processed_tiles, batch_size = 4, shuffle = True)
    return processed_tiles, loader