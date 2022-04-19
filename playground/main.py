# python peripherals
import os
import sys

# numpy

# pandas

# ipython
from IPython.display import display, HTML

# matplotlib
import matplotlib.pyplot as plt

# plotly

# pytorch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms

# wsi-mil
sys.path.insert(1, os.path.join(sys.path[0], './..'))
from nn import datasets
from nn import trainers

# openslide

if __name__ == '__main__':
    dataset_name = 'TCGA'
    tile_size = 256
    desired_magnification = 10
    minimal_tiles_count = 10
    test_fold = 1
    datasets_base_dir_path = f'D:\Pathology'
    negative_examples_count = 2
    dataset_size = 50
    batch_size = 4
    buffers_base_dir = 'C:/GitHub/WSI_MIL/buffers'

    train_dataset = datasets.WSITuplesGenerator(
        dataset_size=dataset_size,
        buffer_size=50,
        replace=True,
        num_workers=1,
        dataset_name=dataset_name,
        tile_size=tile_size,
        desired_magnification=desired_magnification,
        minimal_tiles_count=minimal_tiles_count,
        test_fold=test_fold,
        train=True,
        datasets_base_dir_path=datasets_base_dir_path,
        max_size=50,
        inner_radius=2,
        outer_radius=11)

    train_dataset.start(load_buffer=True, buffer_base_dir_path=buffers_base_dir)

    trainer = trainers.WSIDistanceModelTrainerTest()
    indices = list(range(dataset_size))
    sampler = SubsetRandomSampler(indices)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
    for batch_index, batch_data in enumerate(data_loader, 0):
        preprocessed_input_features = trainer._preprocess_batch(batch_data=batch_data)
        output = preprocessed_input_features.reshape([batch_data['input_features'].shape[0], batch_data['input_features'].shape[1], 3, 256, 256])

        display(HTML(f'<H1>Batch {batch_index}'))

        for tuple_index in range(output.shape[0]):
            fig, axes = plt.subplots(nrows=1, ncols=2 + negative_examples_count, figsize=(10, 20))

            anchor_pic = transforms.ToPILImage()(output[tuple_index, 0, :, :, :])
            positive_pic = transforms.ToPILImage()(output[tuple_index, 1, :, :, :])

            axes[0].imshow(anchor_pic)
            axes[0].axis('off')
            axes[0].set_title('Anchor Tile')
            axes[1].imshow(positive_pic)
            axes[1].axis('off')
            axes[1].set_title('Positive Tile')

            for i in range(2, negative_examples_count + 2):
                negative_pic = transforms.ToPILImage()(output[tuple_index, i, :, :, :])
                axes[i].imshow(negative_pic)
                axes[i].axis('off')
                axes[i].set_title('Negative Tile')

            plt.show()

    train_dataset.stop()
