from pathlib import Path
from tum import TUMDataset
import argh
import shutil
from tqdm import tqdm


def main(data_source: Path):
    dataset = TUMDataset(data_source)
    depth_folder = Path(data_source) / "depth_sync"
    depth_folder.mkdir()
    rgb_folder = Path(data_source) / "rgb_sync"
    rgb_folder.mkdir()

    for paths in tqdm(dataset):
        rgb, depth = paths
        shutil.copy(rgb, rgb_folder)
        shutil.copy(depth, depth_folder)
        #  print(rgb, depth)


if __name__ == "__main__":
    argh.dispatch_command(main)
