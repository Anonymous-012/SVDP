from .builder import DATASETS
from .cityscapes import CityscapesDataset
import re

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger

@DATASETS.register_module()
class FoggyCityscapesDataset(CityscapesDataset):
    """FoggyCityscapesDataset dataset.
        
    """

    def __init__(self, **kwargs):
        self.repl_suffix = kwargs.pop('repl_suffix', r"_leftImg8bit_.*\.png")
        super().__init__(
            img_suffix="0.01.png",
            seg_map_suffix="_gtFine_labelTrainIds.png",
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    # seg_map = img.replace(img_suffix, seg_map_suffix)
                    seg_map = re.sub(self.repl_suffix, seg_map_suffix, img)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos