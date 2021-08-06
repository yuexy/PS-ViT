import os

from PIL import Image
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        label_file = open(meta_file, 'r')
        lines = label_file.readlines()
        label_file.close()
        
        self.num = len(lines)
        self.metas = []
        for line in lines:
            img_path, cls_label = line.rstrip().split()
            self.metas.append((img_path, int(cls_label)))
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        img_name, cls_label = self.metas[idx]
        
        img_path = os.path.join(self.root_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        
        # transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, cls_label
    
    def filename(self, index, basename=False, absolute=False):
        filename = self.metas[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root_dir)
        return filename
    
    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.metas]
