# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset


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
            ├── exp/
            │   ├── train_id.txt  # 逗号分隔ID，如0001,0002,0003
            │   ├── val_id.txt
            │   └── test_id.txt
    """
    # 核心配置抽离，便于维护和扩展
    DATASET_NAME = 'CMshipReID'
    MODALITIES = ['VIS', 'NIR', 'TIR']  # 模态名称与目录名对应
    MOD2CAMID = {'VIS': 0, 'NIR': 1, 'TIR': 2}  # 模态映射为camid
    SUPPORTED_EXTS = ['*.tif', '*.png', '*.jpg', '*.jpeg']  # 支持的图片格式

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        """
        Args:
            root: 数据集根目录
            pid_begin: PID起始偏移量（用于多数据集拼接）
            verbose: 是否打印数据集统计信息
        """
        super(CMship, self).__init__()
        # 路径拼接：兼容Windows/Linux，规范化路径
        self.dataset_dir = osp.abspath(osp.join(root, self.DATASET_NAME))
        self.pid_begin = pid_begin

# 定义ID列表文件路径（解耦硬编码）
        self.train_ids_path = osp.join(self.dataset_dir, 'exp', 'train_id.txt')
        self.val_ids_path = osp.join(self.dataset_dir, 'exp', 'val_id.txt')
        self.test_ids_path = osp.join(self.dataset_dir, 'exp', 'test_id.txt')

        # 加载训练/验证/测试的ID列表
        self.train_ids = self._load_ids(self.train_ids_path)
        self.val_ids = self._load_ids(self.val_ids_path)
        self.test_ids = self._load_ids(self.test_ids_path)

        # 前置检查：避免后续处理出错
        self._check_before_run()

        # 处理数据集（训练集含成对数据，验证/测试集为单模态）
        train, train_pair = self._process_dir_train(relabel=True)
        val = self._process_dir(mode='val', relabel=False)
        test = self._process_dir(mode='test', relabel=False)

        # 打印统计信息（可选）
        if verbose:
            self._print_dataset_stats(train, val, test, train_pair)

        # 赋值数据集核心属性
        self.train = train
        self.train_pair = train_pair
        self.val = val
        self.test = test

        # 统计数据集核心信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)
        self.num_val_pids, self.num_val_imgs, self.num_val_cams, self.num_val_vids = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(self.test)



    def _load_ids(self, file_path):
        """
        加载ID列表文件（逗号分隔，如 0001,0002,0003）
        Args:
            file_path: ID列表文件路径
        Returns:
            list: 格式化后的4位字符串ID列表（如['0001', '0002']）
        """
        if not osp.exists(file_path):
            raise RuntimeError(f"ID列表文件不存在: '{file_path}'")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 空文件处理
        if not content:
            raise RuntimeError(f"ID列表文件为空: '{file_path}'")
        
        # 分割+清洗：去除空字符串、空格，转换为4位格式化ID
        ids_str_list = [s.strip() for s in content.split(',') if s.strip()]
        try:
            formatted_ids = [f"{int(pid_str):03d}" for pid_str in ids_str_list]
        except ValueError as e:
            raise RuntimeError(f"ID列表格式错误（需为数字）: {file_path}, 错误: {e}")
        
        return formatted_ids



    def _check_before_run(self):
        """
        前置检查：确保数据集目录、模态目录、ID列表文件都存在
        """
        # 检查数据集根目录
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"数据集根目录不存在: '{self.dataset_dir}'")
        
        # 检查各模态目录
        for mod in self.MODALITIES:
            mod_dir = osp.join(self.dataset_dir, mod)
            if not osp.exists(mod_dir):
                raise RuntimeError(f"模态目录不存在: '{mod_dir}'")
        
        # 检查ID列表文件
        id_files = [self.train_ids_path, self.val_ids_path, self.test_ids_path]
        for file_path in id_files:
            if not osp.exists(file_path):
                raise RuntimeError(f"ID列表文件不存在: '{file_path}'")


    def _get_img_paths_by_mode(self, mode):
        """
        根据模式（train/val/test）获取对应ID的所有图片路径
        Args:
            mode: 数据集模式（train/val/test）
        Returns:
            list: 排序后的图片绝对路径列表
        """
        # 映射模式到目标ID列表
        mode2ids = {
            'train': self.train_ids,
            'val': self.val_ids,
            'test': self.test_ids
        }
        if mode not in mode2ids:
            raise ValueError(f"无效模式: {mode}，仅支持 train/val/test")
        target_ids = mode2ids[mode]

        img_paths = []
        # 遍历所有模态+所有目标ID，收集图片路径
        for mod in self.MODALITIES:
            mod_dir = osp.join(self.dataset_dir, mod)
            for pid_str in target_ids:
                pid_dir = osp.join(mod_dir, pid_str)
                if not osp.exists(pid_dir):
                    continue  # 该ID无对应目录，跳过
                # 收集该ID下所有支持格式的图片
                for ext in self.SUPPORTED_EXTS:
                    img_paths.extend(glob.glob(osp.join(pid_dir, ext)))

        # 排序保证确定性（避免不同系统遍历顺序不一致）
        return sorted(img_paths)


    def _process_dir(self, mode, relabel=False):
        """
        处理验证/测试集（单模态数据）
        Args:
            mode: 数据集模式（val/test）
            relabel: 是否重标记PID（仅训练集需要）
        Returns:
            list: 数据集列表，格式为 (img_path, pid, camid, trackid)
        """
        img_paths = self._get_img_paths_by_mode(mode)
        if not img_paths:
            print(f"警告: {mode}集未找到任何图片")
            return []

        # 收集所有PID（用于重标记）
        pid_container = set()
        for img_path in img_paths:
            # 兼容Windows(\)和Linux(/)路径：提取PID（倒数第二个目录）
            pid_str = osp.basename(osp.dirname(img_path))
            pid = int(pid_str)
            pid_container.add(pid)

        # PID重标记映射（仅训练集启用）
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # 构建数据集
        dataset = []
        for img_path in img_paths:
            # 提取PID和模态
            pid_str = osp.basename(osp.dirname(img_path))
            pid = int(pid_str)
            mod = osp.basename(osp.dirname(osp.dirname(img_path)))  # 提取模态（VIS/IR）
            camid = self.MOD2CAMID.get(mod, 1)  # 兼容未知模态（默认IR）

            # 重标记PID
            if relabel:
                pid = pid2label[pid]

            # 兼容ReID框架格式：(路径, 偏移后PID, 模态camid, trackid)
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset


    def _process_dir_train(self, relabel=False):
        """
        处理训练集（含VIS与IR成对数据）
        Args:
            relabel: 是否重标记PID
        Returns:
            tuple: (基础训练集, 成对训练集)
        """
        img_paths = self._get_img_paths_by_mode('train')
        if not img_paths:
            return [], []

        # 1. 构建模态-PID-路径映射（便于配对）
        pid_container = set()
        mod2pid2paths = {mod: {} for mod in self.MODALITIES}
        for img_path in img_paths:
            pid_str = osp.basename(osp.dirname(img_path))
            pid = int(pid_str)
            mod = osp.basename(osp.dirname(osp.dirname(img_path)))
            pid_container.add(pid)

            # 初始化PID路径列表
            if pid not in mod2pid2paths[mod]:
                mod2pid2paths[mod][pid] = []
            mod2pid2paths[mod][pid].append(img_path)

        # 2. PID重标记
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # 3. 构建基础训练集（所有图片）
        train_dataset = []
        for img_path in img_paths:
            pid_str = osp.basename(osp.dirname(img_path))
            pid = int(pid_str)
            mod = osp.basename(osp.dirname(osp.dirname(img_path)))
            camid = self.MOD2CAMID.get(mod, 1)

            if relabel:
                pid = pid2label[pid]

            train_dataset.append((img_path, self.pid_begin + pid, camid, 1))

        # 4. 构建成对数据集（VIS为基准，匹配NIR/TIR）
        pair_dataset = []
        vis_pid_paths = mod2pid2paths['VIS']  # VIS模态的所有PID路径
        
        for pid in vis_pid_paths:
            if pid not in mod2pid2paths['NIR'] and pid not in mod2pid2paths['TIR']:
                continue  # 无配对的NIR/TIR，跳过
            

            # 遍历该PID的所有VIS图片，匹配对应IR图片
            for vis_path in vis_pid_paths[pid]:
                vis_pid_label = pid2label[pid] if relabel else pid
                vis_item = (vis_path, self.pid_begin + vis_pid_label, self.MOD2CAMID['VIS'], 1)

                # 匹配NIR
                if pid in mod2pid2paths['NIR']:
                    for nir_path in mod2pid2paths['NIR'][pid]:
                        nir_pid_label = pid2label[pid] if relabel else pid
                        nir_item = (nir_path, self.pid_begin + nir_pid_label, self.MOD2CAMID['NIR'], 1)
                        pair_dataset.append([vis_item, nir_item])
                
                # 匹配TIR
                if pid in mod2pid2paths['TIR']:
                    for tir_path in mod2pid2paths['TIR'][pid]:
                        tir_pid_label = pid2label[pid] if relabel else pid
                        tir_item = (tir_path, self.pid_begin + tir_pid_label,self.MOD2CAMID['TIR'], 1)
                        pair_dataset.append([vis_item, tir_item])

        return train_dataset, pair_dataset


    def get_imagedata_info_pair(self, data):
        """
        统计成对数据集信息（复用原逻辑，增强鲁棒性）
        Args:
            data: 成对数据集列表
        Returns:
            tuple: (num_pids, num_imgs, num_cams, num_tracks)
        """
        if not data:
            return 0, 0, 0, 0

        pids, cams, tracks = [], [], []
        for pair in data:
            for _, pid, camid, trackid in pair:
                pids.append(pid)
                cams.append(camid)
                tracks.append(trackid)

        return len(set(pids)), len(data), len(set(cams)), len(set(tracks))


    def _print_dataset_stats(self, train, val, test, train_pair):
        """
        标准化打印数据集统计信息（增强可读性）
        """
        print("=" * 60)
        print(f"=> CMship ReID Dataset 加载完成")
        print("-" * 60)
        # 基础数据集统计
        self.print_dataset_statistics(train, val, test)
        # 成对数据统计
        if train_pair:
            pair_pids, pair_imgs, pair_cams, _ = self.get_imagedata_info_pair(train_pair)
            print(f"训练集成对数据统计:")
            print(f"  成对PID数: {pair_pids} | 成对样本数: {pair_imgs} | 模态数: {pair_cams}")
        print("=" * 60)



    
# 示例：实例化数据集（放到main中避免导入时执行）
if __name__ == '__main__':
    # Windows路径建议用原始字符串，Linux直接用绝对路径
    dataset = CMship(
        root=r'E:\博士\科研\数据集\跨模态舰船重识别',
        verbose=True,
        pid_begin=0
    )
    # 打印核心信息
    print(f"训练集样本数: {len(dataset.train)}")
    print(f"训练集成对样本数: {len(dataset.train_pair)}")
    print(f"验证集样本数: {len(dataset.val)}")
    print(f"测试集样本数: {len(dataset.test)}")