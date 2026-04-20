# encoding: utf-8
import glob
import os.path as osp
from bases import BaseImageDataset


class KT_Boat(BaseImageDataset):
    """
    CMship dataset (Cross-Modal Ship ReID)
    Modalities: VIS(0), IR(1)
    Dataset structure:
        KT_Boat/
            ├── VIS/
            │   ├── 0001/
            │   ├── 0002/
            │   └── ...
            ├── IR/
    """
    dataset_dir = 'KT_Boat'
    modalities = ['VIS', 'IR']  # 模态名称与目录名对应
    mod2camid = {'VIS': 0, 'IR': 1}  # 模态映射为camid


    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        """
        Args:
            root: 数据集根目录
            pid_begin: PID起始偏移量
            train_ids_path: 训练集ID列表文件路径（每行一个ID，如0001）
            val_ids_path: 验证集ID列表文件路径
            test_ids_path: 测试集ID列表文件路径
        """
        super(KT_Boat, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_ids_path =  osp.join(self.dataset_dir, 'exp', 'train_id.txt')
        val_ids_path =  osp.join(self.dataset_dir, 'exp', 'val_id.txt')
        test_ids_path =  osp.join(self.dataset_dir, 'exp', 'test_id.txt')
        self.pid_begin = pid_begin
        
        # 加载训练/验证/测试的ID列表
        self.train_ids = self._load_ids(train_ids_path)
        self.val_ids = self._load_ids(val_ids_path)
        self.test_ids = self._load_ids(test_ids_path)

        # 检查目录是否存在
        self._check_before_run()

        # 处理训练集（含成对数据）、查询集、图库集
        train, train_pair = self._process_dir_train(relabel=True)
        val = self._process_dir(mode='val', relabel=False)
        test = self._process_dir(mode='test', relabel=False)

        if verbose:
            print("=> KT_Boat ReID Dataset loaded")
            self.print_dataset_statistics(train, val, test)
            if train_pair is not None:
                print("Number of VIS-IR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.val = val
        self.test = test

        # 统计数据信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)
        self.num_val_pids, self.num_val_imgs, self.num_val_cams, self.num_val_vids = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(self.test)



    def _load_ids(self, file_path):
        """加载ID列表文件（逗号分隔，如 0001,0002,0003）"""
        if not osp.exists(file_path):
            raise RuntimeError(f"'{file_path}' ID列表文件不存在")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # 读取全部内容
        
        # 按逗号分割，并去除空字符串、空格
        ids_str_list = [s.strip() for s in content.split(',') if s.strip()]
        
        # 转换为整数ID（如'1'→001）
        return [f"{int(pid_str):04d}" for pid_str in ids_str_list]



    def _check_before_run(self):
        """检查数据集目录和模态目录是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' 数据集根目录不存在")
        for mod in self.modalities:
            mod_dir = osp.join(self.dataset_dir, mod)
            if not osp.exists(mod_dir):
                raise RuntimeError(f"'{mod_dir}' {mod}模态目录不存在")


    def _get_img_paths_by_mode(self, mode):
        """根据模式（train/val/test）获取对应ID的所有图片路径"""
        if mode == 'train':
            target_ids = self.train_ids
        elif mode == 'val':
            target_ids = self.val_ids
        elif mode == 'test':
            target_ids = self.test_ids
        else:
            raise ValueError(f"无效模式: {mode}，仅支持train/val/test")
        

        img_paths = []
        for mod in self.modalities:
            mod_dir = osp.join(self.dataset_dir, mod)
            for pid_str in [str(pid) for pid in target_ids]:
                pid_dir = osp.join(mod_dir, pid_str)
                if not osp.exists(pid_dir):
                    continue
                # 匹配该ID目录下的所有图片（支持tif/png/jpg等）
                for ext in ['*.tif', '*.png', '*.jpg', '*.jpeg']:
                    img_paths.extend(glob.glob(osp.join(pid_dir, ext)))
        return sorted(img_paths)


    def _process_dir(self, mode, relabel=False):
        """处理查询集/图库集（单模态数据）"""
        img_paths = self._get_img_paths_by_mode(mode)
        if not img_paths:
            return []

        # 收集所有PID
        pid_container = set()
        for img_path in img_paths:
            # 从路径中提取PID（如.../VIS/0001/xxx.tif → 0001 → 1）
            pid_str = img_path.split('\\')[-2]
            pid = int(pid_str)
            pid_container.add(pid)
        
        # PID重标记（仅训练集需要，查询/图库集保持原PID）
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in img_paths:
            # 提取PID和模态
            pid_str = img_path.split('\\')[-2]
            pid = int(pid_str)
            mod = img_path.split('\\')[-3]  # VIS/IR
            camid = self.mod2camid[mod]  # 模态映射为camid
            
            # 重标记PID（如果需要）
            if relabel:
                pid = pid2label[pid]
            
            # 格式：(img_path, pid_begin+pid, camid, trackid=1)（兼容原HOSS格式）
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset


    def _process_dir_train(self, relabel=False):
        """处理训练集（含VIS与IR成对数据）"""
        img_paths = self._get_img_paths_by_mode('train')
        if not img_paths:
            return [], []

        # 1. 收集所有PID和模态路径映射
        pid_container = set()
        mod2pid2paths = {mod: {} for mod in self.modalities}  # {VIS: {pid: [path1, ...]}, ...}
        
        for img_path in img_paths:
            pid_str = img_path.split('\\')[-2]
            pid = int(pid_str)
            mod = img_path.split('\\')[-3]
            pid_container.add(pid)
            
            if pid not in mod2pid2paths[mod]:
                mod2pid2paths[mod][pid] = []
            mod2pid2paths[mod][pid].append(img_path)
        
        # 2. PID重标记
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # 3. 构建基础训练数据集（所有图片）
        dataset = []
        for img_path in img_paths:
            pid_str = img_path.split('\\')[-2]
            pid = int(pid_str)
            mod = img_path.split('\\')[-3]
            camid = self.mod2camid[mod]
            
            if relabel:
                pid = pid2label[pid]
            
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        # 4. 构建成对数据集（VIS为基准，匹配NIR/TIR）
        dataset_pair = []
        vis_paths = mod2pid2paths['VIS']  # VIS模态的所有PID路径
        
        for pid in vis_paths:
            if pid not in mod2pid2paths['IR']:
                continue  # 无配对的IR，跳过
            
            # 遍历该PID的所有VIS图片
            for vis_path in vis_paths[pid]:
                vis_pid_label = pid2label[pid] if relabel else pid
                vis_camid = self.mod2camid['VIS']
                vis_item = (vis_path, self.pid_begin + vis_pid_label, vis_camid, 1)
                
                # 匹配IR
                if pid in mod2pid2paths['IR']:
                    for ir_path in mod2pid2paths['IR'][pid]:
                        ir_pid_label = pid2label[pid] if relabel else pid
                        ir_camid = self.mod2camid['IR']
                        ir_item = (ir_path, self.pid_begin + ir_pid_label, ir_camid, 1)
                        dataset_pair.append([vis_item, ir_item])
                

        
        return dataset, dataset_pair


    def get_imagedata_info_pair(self, data):
        """复用原HOSS的成对数据统计逻辑"""
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views
    
# KT_Boat(root = 'E:\博士\科研\数据集\跨模态舰船重识别')
# from .bases import BaseImageDataset
# '/'