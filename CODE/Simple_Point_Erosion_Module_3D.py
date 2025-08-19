import os
import skimage
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
#from monai.inferers import sliding_window_inference

def whether_center_point_is_simple_point(patch):   #( 8, 4)
    """
    The criterion is derived from the following paper:
    
    [1]  X. Hu, “Structure-aware image segmentation with homotopy warping,” Advances in Neural Information Processing Systems, vol. 35, pp.
    24046–24059, 2022.
    [2]  T. Y. Kong and A. Rosenfeld, “Digital topology: Introduction and
        survey,” Computer Vision, Graphics, and Image Processing, vol. 48,
        no. 3, pp. 357–393, 1989.
        
    """

    """ Input:
            patch: a 3x3 binary patch
        Return:
            True: if the center point is a simple point
            False: if the center point is not a simple point
    """
    patch_copy1 = patch.copy()      # used for condition 1
    patch_copy2 = patch.copy()      # used for condition 2

    # （26，6） connectivity
    # count foreground 26 neighbors connected components
    count, num = skimage.measure.label(
        patch_copy1, connectivity=3, return_num=True)   # 26-adjacency connectivity
    # count background 6 neighbors connected components
    patch_copy1 = 1-patch_copy1
    count2, num2 = skimage.measure.label(
        patch_copy1, connectivity=1, return_num=True)   # 6-adjacency connectivity
    
    # flip the center point
    patch_copy2[1, 1, 1] = 1-patch_copy2[1, 1, 1]
    # count foreground 26 neighbors connected components
    count3, num3 = skimage.measure.label(
        patch_copy2, connectivity=3, return_num=True)   # 26-adjacency connectivity
    # count background 4 neighbors connected components
    patch_copy2 = 1-patch_copy2
    count4, num4 = skimage.measure.label(
        patch_copy2, connectivity=1, return_num=True)   # 6-adjacency connectivity
    
    if num == num3 and num2 == num4:        # whether the center point is a simple point
        return True
    else:
        return False
    

def patchify(binary_image):
    B,C,D,H,W = binary_image.shape
    padding=1
    k = 3

    if isinstance(binary_image, torch.Tensor):
        # PyTorch张量填充（格式：(pad_d_before, pad_d_after, pad_h_before, pad_h_after, pad_w_before, pad_w_after)）
        image_paded = torch.nn.functional.pad(
            binary_image, 
            (padding, padding, padding, padding, padding, padding),
            mode='constant', 
            value=0
        )
    patches = image_paded.unfold(2,3,1).unfold(3,3,1).unfold(4,3,1)
    patches = patches.reshape((B*C*D*H*W, 3*3*3))

    return patches


    


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,train_ratio=1,type='train'):
        
        self.total_num = 2**9
        self.train_ratio = train_ratio
        self.train_num = int(self.total_num*train_ratio)
        self.test_num = self.total_num - self.train_num
        self.train_dict_list,self.test_dict_list = self.create_train_dict_list_and_test_dict_list()
        self.type = type
        self.simple_list = []

        if self.type == 'train':
            self.dict_list = self.train_dict_list
        elif self.type == 'test':
            self.dict_list = self.test_dict_list
        else:
            raise ValueError('type must be train or test')
        
    def get_cc_num_and_simple_point_label(self,patch):
        _, num1 = skimage.measure.label(
            patch, connectivity=1, return_num=True)   # 4-adjacency connectivity
        _, num2 = skimage.measure.label(
            patch, connectivity=2, return_num=True)   # 8-adjacency connectivity
        simple_point_label =  None #whether_center_point_is_simple_point(patch)
        return  num1, num2, simple_point_label
    

    # a function to exhaust all the possible patches
    def create_train_dict_list_and_test_dict_list(self):
        total_list = []
        train_dict_list = []
        test_dict_list = []
        simple_count = 0
        # initialize a pytorch lookuptable
        simple_count = 0
        self.simple_list = []
        for i in range(2**9):

            # convert i to binary string and ensure the length of the string is 9
            binary_string = bin(i)[2:].zfill(9)
            # convert the binary string to a list of 0s and 1s
            binary_list = [int(x) for x in binary_string]
            # convert the list of 0s and 1s to a 3x3 array
            patch = np.array(binary_list).reshape(3,3)     
            criterion = whether_center_point_is_simple_point(patch)

            if criterion:
                simple_count = simple_count + 1
                self.simple_list.append(patch)

            total_list.append({'patch':patch,'num1':0,'num2':0,'simple_point_label':criterion})
        # chech no patch are same
        for i in range(len(total_list)):
            for j in range(i+1,len(total_list)):
                if np.array_equal(total_list[i]['patch'],total_list[j]['patch']):
                    raise ValueError('two patches are same')

        
        # shuffle the total_list
        np.random.shuffle(total_list)
        train_dict_list = total_list[:self.train_num]
        test_dict_list = total_list[self.train_num:]

        # numpy save self.simple_list 
        np.save('simple_list.npy',self.simple_list)
        return train_dict_list,test_dict_list

    def __len__(self):
        return len(self.dict_list)
    
    def __getitem__(self, idx):
        return self.dict_list[idx]
    
    def get_train_num(self):
        return self.train_num
    
    def get_test_num(self):
        return self.test_num

class SimplePointProcessor:
    def __init__(self):
        # 定义3D体素块的对称变换（旋转、反射等）
        #self.symmetries = self._generate_3d_symmetries()

        # 中心点在3x3x3体素中的索引（第13位，0-based）
        self.center_index = 13  # (1,1,1)在27个位置中的索引

        # 三维体素所有可能情况2**27
        self.total_num = 2**27

    def _generate_3d_symmetries(self):
        """生成3D体素块的所有对称变换函数"""
        symmetries = []
        
        # 坐标轴排列组合（6种）
        axes_permutations = [
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0)
        ]
        
        # 坐标轴方向（每个轴可正可负，2^3=8种）
        for perm in axes_permutations:
            for flip_x in [False, True]:
                for flip_y in [False, True]:
                    for flip_z in [False, True]:
                        def create_transform(perm, flip_x, flip_y, flip_z):
                            def transform(coords):
                                x, y, z = coords
                                # 应用坐标轴排列
                                x, y, z = [x, y, z][perm[0]], [x, y, z][perm[1]], [x, y, z][perm[2]]
                                # 应用翻转
                                if flip_x: x = 2 - x
                                if flip_y: y = 2 - y
                                if flip_z: z = 2 - z
                                return (x, y, z)
                            return transform
                        symmetries.append(create_transform(perm, flip_x, flip_y, flip_z))
        
        # 去重（6*8=48种独特对称变换）
        unique_symmetries = []
        seen = set()
        for sym in symmetries:
            key = tuple(sym((x, y, z)) for x in range(3) for y in range(3) for z in range(3))
            if key not in seen:
                seen.add(key)
                unique_symmetries.append(sym)
        return unique_symmetries

    def _get_index_from_coords(self, x, y, z):
        """将3D坐标转换为27位中的索引位置"""
        return x * 9 + y * 3 + z

    def _get_coords_from_index(self, idx):
        """将27位中的索引位置转换为3D坐标"""
        x = idx // 9
        remainder = idx % 9
        y = remainder // 3
        z = remainder % 3
        return (x, y, z)

    def _is_minimal_representation(self, i):
        """检查当前体素块是否为其对称等价类中的最小表示"""
        original_coords_list = [self._get_coords_from_index(idx) for idx in range(27)]
        current_min = i
        
        for sym in self.symmetries:
            # 对每个对称变换，计算变换后的索引值
            transformed_bits = 0
            for idx in range(27):
                x, y, z = original_coords_list[idx]
                tx, ty, tz = sym((x, y, z))
                t_idx = self._get_index_from_coords(tx, ty, tz)
                # 提取原始位值并设置到变换后的位置
                if (i >> idx) & 1:
                    transformed_bits |= (1 << t_idx)
            
            if transformed_bits < current_min:
                return False
        return True

    def create_embedding(self, save_dir):
        total_processed = 0
        simple_count = 0
        valid_indices = []  # 存储符合条件的体素块索引
        labels = []         # 存储对应的简单点标签
        simple_indices = []

        self.embedding = nn.Embedding(2**27, 1)
        self.embedding.weight.data.fill_(0.0)
        # 只处理中心点为1的体素块（第13位为1）
        # 计算范围：2^13 ~ 2^27 - 1，且第13位为1
        start = 1 << self.center_index
        end = (1 << 27) - 1

        for i in tqdm(range(start, end + 1),desc="Processing 3*3*3 volume"):
            # 确保中心点为1（冗余检查，防止范围计算错误）
            if not (i & (1 << self.center_index)):
                continue
            
            # 检查是否为对称等价类中的最小表示
            # if not self._is_minimal_representation(i):
            #     continue
            
            # 生成体素块
            patch = np.zeros((3, 3, 3), dtype=np.uint8)
            for idx in range(27):
                x, y, z = self._get_coords_from_index(idx)
                patch[x, y, z] = (i >> idx) & 1
            
            # 判断是否为简单点
            is_simple = whether_center_point_is_simple_point(patch)
            # valid_indices.append(i)
            labels.append(is_simple)
            
            if is_simple:
                simple_count += 1
                simple_indices.append(i)
            
            total_processed += 1

        if simple_indices:
            simple_indices = torch.tensor(simple_indices, dtype=torch.long)
            self.embedding.weight.data[simple_indices] = 1.0
        
            torch.save(self.embedding.state_dict(), os.path.join(save_dir, 'simple_lookup_embedding.pt'))

        # 保存简单点列表
        #np.save('simple_list_3D.npy', self.simple_list)

        print(f"总处理体素块数量: {total_processed}")
        print(f"简单点数量: {simple_count}")
        print(f"Embedding path:{os.path.join(save_dir, 'simple_lookup_embedding.pt')}")
    
class Simple_Point_Erosion_module():
    def __init__(self,target_D_H_W=(960,960,960),device = torch.device("cuda:0"), embd_path=None, save_path=None) -> None:
        if embd_path == None:
            if save_path is None:
                save_path = "./SP"
            print("Generate New 3D pattern embeddings")
            generater = SimplePointProcessor()
            generater.create_embedding(save_path)

            embd_path = os.path.join(save_path, 'simple_lookup_embedding.pt')
        
        
        #self.simple_point_dataset = PatchDataset()
        # self.train_num = self.patch_dataset.get_train_num()
        self.D,self.H,self.W = target_D_H_W
        self.device = device
        #self.Construct_a_LookupTable_of_SimplePoints()

        self.SPLT = nn.Embedding(2**27,1)
        self.SPLT.load_state_dict(torch.load(embd_path, map_location=torch.device('cpu')))
        self.SPLT = self.SPLT.to(device)
        self.lookup_table = self.SPLT

        self.build_order_masks()

    # def Construct_a_LookupTable_of_SimplePoints(self): # build a simple point lookup table
    #     # creata a
    #     self.SPLT = nn.Embedding(2**9,1)
    #     # initialize all the value to 0
        
    #     self.SPLT.weight.data.fill_(0)
    #     for data_dict in self.simple_point_dataset:
    #         patch = data_dict['patch']
    #         num1 = data_dict['num1']
    #         num2 = data_dict['num2']
    #         simple_point_label = data_dict['simple_point_label']
    #         if simple_point_label:
    #             # convert patch to a 9 bit number
    #             patch = patch.flatten()
    #             patch = patch.dot(2**np.arange(8, -1, -1))
    #             self.SPLT.weight.data[patch] = 1.0
    #     self.SPLT = self.SPLT.to(self.device)
    #     self.lookup_table = self.SPLT
    
    # def build_order_masks(self):  # this method could be replaced with a more efficient way
    #     self.order_mask_list = []
    #     m1 = torch.zeros(self.H,self.W)
    #     m2 = torch.zeros(self.H,self.W)
    #     m3 = torch.zeros(self.H,self.W)
    #     m4 = torch.zeros(self.H,self.W)
    #     for i in range(self.H):
    #         for j in range(self.W):
    #             i_mod2 = i%2
    #             j_mod2 = j%2
    #             if i_mod2 == 0 and j_mod2 == 0:
    #                 m1[i,j] = 1
    #             elif i_mod2 == 0 and j_mod2 == 1:
    #                 m2[i,j] = 1
    #             elif i_mod2 == 1 and j_mod2 == 0:
    #                 m3[i,j] = 1
    #             elif i_mod2 == 1 and j_mod2 == 1:
    #                 m4[i,j] = 1
    #     self.order_mask_list.append(m1.unsqueeze(0).unsqueeze(0).to(self.device))
    #     self.order_mask_list.append(m2.unsqueeze(0).unsqueeze(0).to(self.device))
    #     self.order_mask_list.append(m3.unsqueeze(0).unsqueeze(0).to(self.device))
    #     self.order_mask_list.append(m4.unsqueeze(0).unsqueeze(0).to(self.device))
    
    def build_order_masks(self):
        """
        生成8个(128,128,128)的3D掩码，每个掩码中1值体素的3×3×3邻域内全为0
        
        返回:
            list: 包含8个numpy数组的列表，每个数组形状为(128,128,128)
        """
        # 初始化8个掩码，全部为0
        self.order_mask_list = [torch.zeros((self.D, self.H, self.W)) for _ in range(8)]
        #order_mask_list = [torch.zeros((128,128,128)) for _ in range(8)]
        
        # 为每个掩码设置1值体素的位置模式
        # 我们将空间在三个维度上各分成2份，形成8个区域
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # 计算当前掩码索引
                    mask_idx = i * 4 + j * 2 + k
                    
                    # 确定当前区域在三个维度上的起始和步长
                    # 步长设为3，确保3×3×3邻域内不会有其他1值
                    x = slice(i, self.D, 2)
                    y = slice(j, self.H, 2)
                    z = slice(k, self.W, 2)
                    
                    # 在当前区域设置1值
                    self.order_mask_list[mask_idx][x, y, z] = 1
                    #order_mask_list[mask_idx][x, y, z] = 1
        for i in range(8):
            self.order_mask_list[i] = self.order_mask_list[i].unsqueeze(0).unsqueeze(0).to(self.device)

           
    def lookup_in_SPLT(self, x):
        patchfied_image = patchify(x)
        # shape of patchfied_image: B*D*H*W, 27
        
        #patchfied_image = patchfied_image.double()
        num, length = patchfied_image.shape
        # device = patchfied_image.device

        patchfied_image = patchfied_image.to(torch.int32)
        result = torch.zeros(num, device=patchfied_image.device, dtype=torch.int32)
        shifts = torch.arange(26, -1, -1, dtype=torch.int32, device=patchfied_image.device)

        for i in range(27):
            # 提取第i通道的二进制值（0或1）
            bit = patchfied_image[:, i].to(torch.int64)  # 转为int64避免移位溢出
            # 移位并累加到结果（1 << shifts[i] 等价于 2^shifts[i]）
            result += bit * (1 << shifts[i])
        
        #patchfied_image = patchfied_image.matmul(2**torch.arange(26, -1, -1, device=patchfied_image.device).to(torch.int32))
        #result = result.to(device=device)
        #patchfied_image = patchfied_image.to(device='cuda')
        embeddings = self.lookup_table(result)
        
        return embeddings
    
             
    def PSPC(self,T,M_T,i):
        
        # T shape: B, 1, H, W
        assert T.shape == M_T.shape
        if T.shape[-3:] == (self.D,self.H,self.W):
            pass
        else:
            self.D,self.H,self.W = T.shape[-3:]
            self.build_order_masks()
            print("rebuild order masks to fit the shape of T, new shape: ",T.shape)
            
        picked_order_mask = self.order_mask_list[i]
        
        B,_,D,H,W = T.shape
        
        global_simple_point_label = self.lookup_in_SPLT(T)
        # shape of global_simple_point_label: B*H*W, 1
        global_simple_point_label = global_simple_point_label.reshape(B,1,D,H,W)
        global_simple_point_label = global_simple_point_label * picked_order_mask
        
        return global_simple_point_label
    
    @torch.no_grad()
    def RSPE(self,T,M_T,max_K=1000):
        T = T.to(self.device)
        M_T = M_T.to(self.device)
        k=0
        intermidiate_T = []
        last_sum_of_T = T.sum()
        while True:
            intermidiate_T.append(T)
            # print(k)
            k+=1
            S1 = self.PSPC(T,M_T,0)
            T = T - S1 * T * M_T
            S2 = self.PSPC(T,M_T,1)
            T = T - S2 * T * M_T
            S3 = self.PSPC(T,M_T,2)
            T = T - S3 * T * M_T
            S4 = self.PSPC(T,M_T,3)
            T = T - S4 * T * M_T
            if k>=max_K:            # termination condition 1: the number of iterations reaches the maximum
                break
            sum_of_T = T.sum()
            if sum_of_T == last_sum_of_T:       # termination condition 2: the sum of T does not change
                break
            last_sum_of_T = sum_of_T
            
        return T,intermidiate_T
    
    
    loop_forward = RSPE



    
    
    
    

