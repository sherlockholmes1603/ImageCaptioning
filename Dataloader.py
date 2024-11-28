
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
from utils import Vocabulary, CaptionCollate

class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, caption_delimiter='|', 
                 image_column='image_name', text_column='caption_text',
                 transform=None, freq_threshold=5,
                 train=True, split_val=0.2):
        self.root_dir = root_dir
        self.caption_file = caption_file
        self.caption_delimiter = caption_delimiter
        self.image_column = image_column
        self.text_column = text_column
        
        self.dataframe = pd.read_csv(caption_file, delimiter=caption_delimiter)
        self.transform = transform
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.dataframe[self.text_column].tolist())
        
        self.train = train 
        self.split_val = split_val
        self._do_split_train_valid()
        
    def _do_split_train_valid(self):
        imgs_train, imgs_valid, caps_train, caps_valid = train_test_split(
            self.dataframe[self.image_column], self.dataframe[self.text_column], 
            test_size=self.split_val, random_state=16
        )
        
        if self.train:
            self.imgs = imgs_train
            self.captions = caps_train
        else:
            self.imgs = imgs_valid
            self.captions = caps_valid
            
        self.imgs = self.imgs.tolist()
        self.captions = self.captions.tolist()
        

    def __len__(self):
        return len(self.imgs)
    
    def _numericalized_caption(self, caption):
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return numericalized_caption

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        ncaption = self._numericalized_caption(caption)

        return img, torch.tensor(ncaption)

def flickr8k_dataloader(root_folder, caption_file, transform, train=True,
                        batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    
    dataset = FlickrDataset(root_folder, caption_file, transform=transform, train=train)
    PAD_IDX = dataset.vocab.stoi["<PAD>"]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=shuffle, pin_memory=pin_memory, 
                            collate_fn=CaptionCollate(pad_idx=PAD_IDX))
    
    return dataloader, dataset
