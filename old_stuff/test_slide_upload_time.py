import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

dset = datasets.WSI_REGdataset(DataSet='ABCTB',
                               tile_size=256,
                               target_kind='ER',
                               test_fold=2,
                               print_timing=True,
                               transform_type='rvf',
                               n_patches=10,
                               mag=10)
dset_loader = DataLoader(dset, batch_size=10)


writer = all_writer = SummaryWriter(os.path.join('../runs', 'time_measurements', dset.DataSet))

for batch_idx, (_, _, time_list, _, _) in enumerate(tqdm(dset_loader)):
    open_slide, extract_tiles, aug_time, total_time = time_list
    extract_tiles_mean = extract_tiles.cpu().detach().numpy().mean()
    aug_time_mean = aug_time.cpu().detach().numpy().mean()
    total_time_mean = total_time.cpu().detach().numpy().mean()

    writer.add_scalar('Tile Extraction Mean Time', extract_tiles_mean, batch_idx)
    writer.add_scalar('Tile Augmentation Mean Time', aug_time_mean, batch_idx)
    writer.add_scalar('Total Mean Time', total_time_mean, batch_idx)


print('Done!')

