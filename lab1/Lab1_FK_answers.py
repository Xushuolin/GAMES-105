import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_stack_list = []
    offset_list = []
    my_joint_dict = {}

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = [name for name in lines[i].split()]
            next_line = [name for name in lines[i+1].split()]
            if line[0] == "HIERARCHY":
                continue
            if line[0] == "MOTION":
                break
            if line[0] == "ROOT" or line[0] == "JOINT":
                joint_name.append(line[-1])
                joint_stack_list.append(line[-1])
            if line[0] == "End":
                joint_name.append(joint_name[-1]+"_end")
                joint_stack_list.append(joint_name[-1])
            if line[0] == "OFFSET":
                offset_list.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == "}":
                joint_index = joint_stack_list.pop()
                if joint_stack_list == []:
                    continue
                else:
                    my_joint_dict[joint_index] = joint_stack_list[-1]
        for i in joint_name:
            if i == "RootJoint":
                joint_parent.append(-1)
            else:
                joint_parent_name = my_joint_dict[i]
                joint_parent.append(joint_name.index(joint_parent_name))
        joint_offset = np.array(offset_list).reshape(-1, 3)
    # print(joint_offset, type(joint_offset), joint_offset.shape)
    print(joint_parent)
    # exit()
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    motion_channels_data = motion_data[frame_id]
    root_position = np.array(motion_channels_data[0:3])
    joint_local_rotation = []
    count = 0
    for i in range(len(joint_name)):
        if '_end' in joint_name[i]:
            joint_local_rotation.append([0., 0., 0.])
        else:
            joint_local_rotation.append(motion_channels_data[3*count+3: 3*count+6])
            count += 1

    # Traverse list, parent node compute R finished before child node, so traverse list from start to end is OK
    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            # Qroot = bvh_channels_get_root_rotation
            # Proot = bvh_channels_get_root_position
            joint_orientation = R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_position = root_position.reshape(1, -1)  # align matrix dimension
        else:
            # Qi = Qi-parent * bvh_channels_get_i_rotation
            # Pi = Pi-parent + offset-i * Oi-parent.T ,  note: Raw Vector * transpose Right Rotation matrix
            joint_orientation = R.from_quat(joint_orientations[joint_parent[i]][0]) * R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_position = joint_positions[joint_parent[i]] + joint_offset[i] * np.asmatrix(R.from_quat(joint_orientations[joint_parent[i]][0]).as_matrix()).transpose()
        joint_positions.append(np.array(joint_position))
        joint_orientations.append(joint_orientation.as_quat().reshape(1, -1))

    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = np.concatenate(joint_orientations, axis=0)
    return joint_positions, joint_orientations
        
    


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_dict = {}
    end_index_A = []
    joint_remove_A = []
    joint_remove_T = []



    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)
    motion_data_A = load_motion_data(A_pose_bvh_path)
    motion_shape_A = motion_data_A.shape

    root_position = motion_data_A[:, :3]
    motion_data_A = motion_data_A[:, 3:]
    motion_data = np.zeros(motion_data_A.shape)
    print(root_position.shape)
    print((motion_data_A[1]).reshape(-1,3))
    # exit()

    for i in joint_name_A:
        if "_end" not in i:
            joint_remove_A.append(i)

    for i in joint_name_T:
        if "_end" not in i:
            joint_remove_T.append(i)


    for index, name in enumerate(joint_remove_A):
        motion_dict[name] = motion_data_A[:, 3*index:3*(index+1)]

    # print(motion_dict)
    # exit()
    for index, name in enumerate(joint_remove_T):
        if name == "lShoulder":
            motion_dict[name][:, 2] -= 45
        elif name == "rShoulder":
            motion_dict[name][:, 2] += 45
        motion_data[:, 3*index:3*(index+1)] = motion_dict[name]
    # print(motion_dict)

    motion_data = np.concatenate([root_position, motion_data], axis=1)
    # print((motion_data[0]).reshape(-1,3))
    #print(motion_data[0])
    # exit()
    return motion_data