# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset


class HOSS(BaseImageDataset):
    """
    HOSS ReID dataset
    Modalities: VIS(0), NIR(1), TIR(2)
    Dataset structure:
        HOSS\\
            ├── bounding_box_train\\
            │   ├── 0001_s05c1_RGB.tif\\
            │   ├── 0001_s05c2_SAR.tif\\
            │   └── ...
            ├── query\\
            ├── bounding_box_test\\
    """
    dataset_dir = 'HOSS'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(HOSS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> HOSS ReID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split('\\')[-1].split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(img_path.split('\\')[-1].split('_')[0])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('\\')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
    
# 示例：实例化数据集（放到main中避免导入时执行）
if __name__ == '__main__':
    # Windows路径建议用原始字符串，Linux直接用绝对路径
    dataset = HOSS(
        root=r'E:\博士\科研\数据集\跨模态舰船重识别',
        verbose=True,
        pid_begin=0
    )
    # 打印核心信息
    print(f"训练集样本数: {len(dataset.train)}")
    print(f"验证集样本数: {len(dataset.query)}")
    print(f"测试集样本数: {len(dataset.gallery)}")