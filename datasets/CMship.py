# encoding: utf-8
import os
import re
import glob
import fnmatch
import os.path as osp
from bases import BaseImageDataset


class CMship(BaseImageDataset):
    """
    CMship dataset (Cross-Modal Ship ReID)
    Modalities: VIS(0), NIR(1), TIR(2)
    Dataset structure:
        CMshipReID/
            ├── VIS/
            │   ├── 0001/
            │   ├── 0002/
            │   └── ...
            ├── NIR/
            ├── TIR/
            └── exp/
                ├── train_id.txt
                ├── val_id.txt
                └── test_id.txt
    """
    MODALITIES = ['VIS', 'NIR', 'TIR']  # 模态名称与目录名对应
    MOD2CAMID = {'VIS': 0, 'NIR': 1, 'TIR': 2}  # 模态映射为camid

    def __init__(self, root='', verbose=True, pid_begin=0, dataset_dir="CMshipReID", 
                 img_exts=None, build_pair=True, **kwargs):
        """
        初始化CMship数据集
        
        Args:
            root (str): 数据集根目录
            verbose (bool): 是否打印数据集统计信息
            pid_begin (int): PID起始偏移量
            dataset_dir (str): 数据集子目录名（默认CMshipReID）
            img_exts (list): 图片后缀列表（默认['*.tif', '*.png', '*.jpg', '*.jpeg']）
            build_pair (bool): 是否构建VIS-NIR/TIR成对数据
            **kwargs: 其他扩展参数
        """
        super(CMship, self).__init__()
        # 跨平台路径拼接
        self.dataset_root = osp.join(root, dataset_dir) if root else dataset_dir
        self.pid_begin = pid_begin
        self.verbose = verbose
        self.build_pair = build_pair
        self.img_exts = img_exts if img_exts is not None else ['*.tif', '*.png', '*.jpg', '*.jpeg']

        # 加载训练/验证/测试的ID列表
        self.train_ids = self._load_ids(osp.join(self.dataset_root, 'exp', 'train_id.txt'))
        self.val_ids = self._load_ids(osp.join(self.dataset_root, 'exp', 'val_id.txt'))
        self.test_ids = self._load_ids(osp.join(self.dataset_root, 'exp', 'test_id.txt'))

        # 检查目录是否存在
        self._check_before_run()

        # 处理训练集、查询集、图库集
        if self.build_pair:
            train, train_pair = self._process_dir_train(relabel=True)
        else:
            train = self._process_dir(mode='train', relabel=True)
            train_pair = []
        val = self._process_dir(mode='val', relabel=False)
        test = self._process_dir(mode='test', relabel=False)

        if self.verbose:
            print("=> CMship ReID Dataset loaded")
            self.print_dataset_statistics(train, val, test)
            if train_pair:
                print(f"Number of VIS-NIR/TIR pair: {len(train_pair)}")
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.val = val
        self.test = test

        # 统计数据信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pair_pids = self.num_train_pair_imgs = self.num_train_pair_cams = self.num_train_pair_vids = 0
        if train_pair:
            self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(train_pair)
        self.num_val_pids, self.num_val_imgs, self.num_val_cams, self.num_val_vids = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(self.test)

    def _load_ids(self, file_path):
        """加载ID列表文件（兼容逗号/换行/空格/制表符分隔）
        
        Args:
            file_path (str): ID列表文件路径
        
        Returns:
            list: 3位数字格式的ID字符串列表，如['001', '002']
        
        Raises:
            RuntimeError: 文件不存在/为空/包含非数字ID
        """
        if not osp.exists(file_path):
            raise RuntimeError(f"'{file_path}' ID列表文件不存在")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # 合并所有行并按多分隔符拆分
        content = ' '.join(lines)
        ids_str_list = [s.strip() for s in re.split(r'[, \t\n]+', content) if s.strip()]
        
        if not ids_str_list:
            raise RuntimeError(f"'{file_path}' ID列表文件为空")
        
        # 转换为3位数字字符串
        try:
            return [f"{int(pid_str):03d}" for pid_str in ids_str_list]
        except ValueError as e:
            raise RuntimeError(f"ID列表文件'{file_path}'包含非数字ID: {e}")

    def _check_before_run(self):
        """检查数据集目录、模态目录、exp目录是否存在"""
        if not osp.exists(self.dataset_root):
            raise RuntimeError(f"'{self.dataset_root}' 数据集根目录不存在")
        
        # 检查模态目录
        for mod in self.MODALITIES:
            mod_dir = osp.join(self.dataset_root, mod)
            if not osp.exists(mod_dir):
                raise RuntimeError(f"'{mod_dir}' {mod}模态目录不存在")
        
        # 检查exp目录（ID列表文件所在）
        exp_dir = osp.join(self.dataset_root, 'exp')
        if not osp.exists(exp_dir):
            raise RuntimeError(f"'{exp_dir}' exp目录（含ID列表）不存在")

    def _get_img_paths_by_mode(self, mode):
        """根据模式（train/val/test）获取对应ID的所有图片路径
        
        Args:
            mode (str): 模式，仅支持train/val/test
        
        Returns:
            list: 按字母序排序的图片路径列表
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"无效模式: {mode}，仅支持train/val/test")
        
        target_ids = getattr(self, f'{mode}_ids')
        img_paths = []
        
        for mod in self.MODALITIES:
            mod_dir = osp.join(self.dataset_root, mod)
            for pid_str in target_ids:
                pid_dir = osp.join(mod_dir, pid_str)
                if not osp.exists(pid_dir):
                    continue
                
                # 批量匹配图片后缀（减少IO操作）
                for file in os.listdir(pid_dir):
                    file_lower = file.lower()
                    if any(fnmatch.fnmatch(file_lower, ext.lower().replace('*.', '')) for ext in self.img_exts):
                        img_paths.append(osp.join(pid_dir, file))
        
        return sorted(img_paths)

    def _process_dir(self, mode, relabel=False):
        """处理查询集/图库集（单模态数据）
        
        Args:
            mode (str): 模式，train/val/test
            relabel (bool): 是否对PID重新标记（训练集建议True）
        
        Returns:
            list: 数据集列表，格式为[(img_path, pid, camid, trackid), ...]
        """
        img_paths = self._get_img_paths_by_mode(mode)
        if not img_paths:
            print(f"警告：{mode}模式下未找到任何图片")
            return []

        # 收集所有PID
        pid_container = set()
        path_info = {}  # 缓存PID和模态，避免重复解析
        for img_path in img_paths:
            pid_str = os.path.basename(os.path.dirname(img_path))
            pid = int(pid_str)
            mod = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            path_info[img_path] = (pid, mod)
            pid_container.add(pid)
        
        # PID重标记
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # 构建数据集
        dataset = []
        for img_path in img_paths:
            pid, mod = path_info[img_path]
            camid = self.MOD2CAMID[mod]
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        return dataset

    def _process_dir_train(self, relabel=False):
        """处理训练集（含VIS与NIR/TIR成对数据）
        
        Args:
            relabel (bool): 是否对PID重新标记
        
        Returns:
            tuple: (dataset, dataset_pair)
                - dataset: 基础训练数据集（所有图片）
                - dataset_pair: VIS-NIR/TIR成对数据集
        """
        img_paths = self._get_img_paths_by_mode('train')
        if not img_paths:
            print("警告：训练集未找到任何图片")
            return [], []

        # 1. 缓存路径解析结果，避免重复计算
        path_info = {}  # {img_path: (pid, mod)}
        pid_container = set()
        mod2pid2paths = {mod: {} for mod in self.MODALITIES}
        
        for img_path in img_paths:
            pid_str = os.path.basename(os.path.dirname(img_path))
            pid = int(pid_str)
            mod = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            path_info[img_path] = (pid, mod)
            pid_container.add(pid)
            
            if pid not in mod2pid2paths[mod]:
                mod2pid2paths[mod][pid] = []
            mod2pid2paths[mod][pid].append(img_path)
        
        # 2. PID重标记
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # 3. 构建基础训练数据集
        dataset = []
        for img_path in img_paths:
            pid, mod = path_info[img_path]
            camid = self.MOD2CAMID[mod]
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        # 4. 构建成对数据集
        dataset_pair = []
        vis_paths = mod2pid2paths.get('VIS', {})
        
        for pid in vis_paths:
            # 无NIR/TIR配对则跳过
            if pid not in mod2pid2paths['NIR'] and pid not in mod2pid2paths['TIR']:
                continue
            
            # 遍历VIS图片
            for vis_path in vis_paths[pid]:
                vis_pid = pid2label[pid] if relabel else pid
                vis_item = (vis_path, self.pid_begin + vis_pid, self.MOD2CAMID['VIS'], 1)
                
                # 匹配NIR
                if pid in mod2pid2paths['NIR']:
                    for nir_path in mod2pid2paths['NIR'][pid]:
                        nir_pid = pid2label[pid] if relabel else pid
                        nir_item = (nir_path, self.pid_begin + nir_pid, self.MOD2CAMID['NIR'], 1)
                        dataset_pair.append([vis_item, nir_item])
                
                # 匹配TIR
                if pid in mod2pid2paths['TIR']:
                    for tir_path in mod2pid2paths['TIR'][pid]:
                        tir_pid = pid2label[pid] if relabel else pid
                        tir_item = (tir_path, self.pid_begin + tir_pid, self.MOD2CAMID['TIR'], 1)
                        dataset_pair.append([vis_item, tir_item])
        
        return dataset, dataset_pair

    def get_imagedata_info_pair(self, data):
        """统计成对数据集的PID/图片/相机/轨迹数量
        
        Args:
            data (list): 成对数据集，格式为 [[(img_path, pid, camid, trackid), ...], ...]
        
        Returns:
            tuple: (num_pids, num_imgs, num_cams, num_views)
                - num_pids: 唯一PID数量
                - num_imgs: 成对图片总数
                - num_cams: 唯一相机ID数量
                - num_views: 唯一轨迹ID数量
        """
        pids, cams, tracks = [], [], []

        for img_pair in data:
            for _, pid, camid, trackid in img_pair:
                pids.append(pid)
                cams.append(camid)
                tracks.append(trackid)
        
        num_pids = len(set(pids))
        num_cams = len(set(cams))
        num_imgs = len(data)
        num_views = len(set(tracks))
        
        return num_pids, num_imgs, num_cams, num_views
    
# dataset = CMship(root = 'E:\博士\科研\数据集\跨模态舰船重识别')
# print(dataset)
# from .bases import BaseImageDataset
# '/'