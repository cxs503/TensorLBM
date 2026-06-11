import torch
import numpy as np
from torch.utils.data import Dataset


def read_train_file1(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file1(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data1(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff1"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file1(old_path1, n_train)
    ux1 = read_train_file1(old_path2, n_train)
    uy1 = read_train_file1(old_path3, n_train)
    uz1 = read_train_file1(old_path4, n_train)
    press2 = read_test_file1(old_path1, n_test)
    ux2 = read_test_file1(old_path2, n_test)
    uy2 = read_test_file1(old_path3, n_test)
    uz2 = read_test_file1(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file2(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file2(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data2(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff1"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file2(old_path1, n_train)
    ux1 = read_train_file2(old_path2, n_train)
    uy1 = read_train_file2(old_path3, n_train)
    uz1 = read_train_file2(old_path4, n_train)
    press2 = read_test_file2(old_path1, n_test)
    ux2 = read_test_file2(old_path2, n_test)
    uy2 = read_test_file2(old_path3, n_test)
    uz2 = read_test_file2(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file7(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(875000, dtype=np.float32)
        for j in range(50):
            res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file7(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = np.empty(875000, dtype=np.float32)
        for j in range(50):
            res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data7(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff4"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file7(old_path1, n_train)
    ux1 = read_train_file7(old_path2, n_train)
    uy1 = read_train_file7(old_path3, n_train)
    uz1 = read_train_file7(old_path4, n_train)
    press2 = read_test_file7(old_path1, n_test)
    ux2 = read_test_file7(old_path2, n_test)
    uy2 = read_test_file7(old_path3, n_test)
    uz2 = read_test_file7(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file8(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file8(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data8(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff4"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file8(old_path1, n_train)
    ux1 = read_train_file8(old_path2, n_train)
    uy1 = read_train_file8(old_path3, n_train)
    uz1 = read_train_file8(old_path4, n_train)
    press2 = read_test_file8(old_path1, n_test)
    ux2 = read_test_file8(old_path2, n_test)
    uy2 = read_test_file8(old_path3, n_test)
    uz2 = read_test_file8(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file11(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(875000, dtype=np.float32)
        for j in range(50):
            res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file11(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(875000, dtype=np.float32)
        for j in range(50):
            res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data11(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff6"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file11(old_path1, n_train)
    ux1 = read_train_file11(old_path2, n_train)
    uy1 = read_train_file11(old_path3, n_train)
    uz1 = read_train_file11(old_path4, n_train)
    press2 = read_test_file11(old_path1, n_test)
    ux2 = read_test_file11(old_path2, n_test)
    uy2 = read_test_file11(old_path3, n_test)
    uz2 = read_test_file11(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file12(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file12(path, num):
    result = np.empty((num, 875000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data12(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff6"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file12(old_path1, n_train)
    ux1 = read_train_file12(old_path2, n_train)
    uy1 = read_train_file12(old_path3, n_train)
    uz1 = read_train_file12(old_path4, n_train)
    press2 = read_test_file12(old_path1, n_test)
    ux2 = read_test_file12(old_path2, n_test)
    uy2 = read_test_file12(old_path3, n_test)
    uz2 = read_test_file12(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file13(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file13(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data13(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff7"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file13(old_path1, n_train)
    ux1 = read_train_file13(old_path2, n_train)
    uy1 = read_train_file13(old_path3, n_train)
    uz1 = read_train_file13(old_path4, n_train)
    press2 = read_test_file13(old_path1, n_test)
    ux2 = read_test_file13(old_path2, n_test)
    uy2 = read_test_file13(old_path3, n_test)
    uz2 = read_test_file13(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file14(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file14(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data14(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff7"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file14(old_path1, n_train)
    ux1 = read_train_file14(old_path2, n_train)
    uy1 = read_train_file14(old_path3, n_train)
    uz1 = read_train_file14(old_path4, n_train)
    press2 = read_test_file14(old_path1, n_test)
    ux2 = read_test_file14(old_path2, n_test)
    uy2 = read_test_file14(old_path3, n_test)
    uz2 = read_test_file14(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file15(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for j in range(200):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file15(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for j in range(200):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data15(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file15(old_path1, n_train)
    ux1 = read_train_file15(old_path2, n_train)
    uy1 = read_train_file15(old_path3, n_train)
    uz1 = read_train_file15(old_path4, n_train)
    press2 = read_test_file15(old_path1, n_test)
    ux2 = read_test_file15(old_path2, n_test)
    uy2 = read_test_file15(old_path3, n_test)
    uz2 = read_test_file15(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file16(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file16(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data16(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file16(old_path1, n_train)
    ux1 = read_train_file16(old_path2, n_train)
    uy1 = read_train_file16(old_path3, n_train)
    uz1 = read_train_file16(old_path4, n_train)
    press2 = read_test_file16(old_path1, n_test)
    ux2 = read_test_file16(old_path2, n_test)
    uy2 = read_test_file16(old_path3, n_test)
    uz2 = read_test_file16(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file17(path, num):
    result = np.empty((num, 1000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[:,:25,:]
        res = np.empty(1000000, dtype=np.float32)
        for i in range(200):
            res[i * 5000: (i + 1) * 5000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file17(path, num):
    result = np.empty((num, 1000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[:, :25, :]
        res = np.empty(1000000, dtype=np.float32)
        for i in range(200):
            res[i * 5000: (i + 1) * 5000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data17(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file17(old_path1, n_train)
    ux1 = read_train_file17(old_path2, n_train)
    uy1 = read_train_file17(old_path3, n_train)
    uz1 = read_train_file17(old_path4, n_train)
    press2 = read_test_file17(old_path1, n_test)
    ux2 = read_test_file17(old_path2, n_test)
    uy2 = read_test_file17(old_path3, n_test)
    uz2 = read_test_file17(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file18(path, num):
    result = np.empty((num, 1000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[:, :25, :]
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file18(path, num):
    result = np.empty((num, 1000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[:, :25, :]
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data18(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file18(old_path1, n_train)
    ux1 = read_train_file18(old_path2, n_train)
    uy1 = read_train_file18(old_path3, n_train)
    uz1 = read_train_file18(old_path4, n_train)
    press2 = read_test_file18(old_path1, n_test)
    ux2 = read_test_file18(old_path2, n_test)
    uy2 = read_test_file18(old_path3, n_test)
    uz2 = read_test_file18(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file23(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for i in range(200):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file23(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for i in range(200):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data23(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff12"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file23(old_path1, n_train)
    ux1 = read_train_file23(old_path2, n_train)
    uy1 = read_train_file23(old_path3, n_train)
    uz1 = read_train_file23(old_path4, n_train)
    press2 = read_test_file23(old_path1, n_test)
    ux2 = read_test_file23(old_path2, n_test)
    uy2 = read_test_file23(old_path3, n_test)
    uz2 = read_test_file23(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file24(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file24(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data24(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff12"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file24(old_path1, n_train)
    ux1 = read_train_file24(old_path2, n_train)
    uy1 = read_train_file24(old_path3, n_train)
    uz1 = read_train_file24(old_path4, n_train)
    press2 = read_test_file24(old_path1, n_test)
    ux2 = read_test_file24(old_path2, n_test)
    uy2 = read_test_file24(old_path3, n_test)
    uz2 = read_test_file24(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file25(path, num):
    result = np.empty((num, 125000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(125000, dtype=np.float32)
        for i in range(50):
            res[i * 2500: (i + 1) * 2500] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file25(path, num):
    result = np.empty((num, 125000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(125000, dtype=np.float32)
        for i in range(50):
            res[i * 2500: (i + 1) * 2500] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data25(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff13"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file25(old_path1, n_train)
    ux1 = read_train_file25(old_path2, n_train)
    uy1 = read_train_file25(old_path3, n_train)
    uz1 = read_train_file25(old_path4, n_train)
    press2 = read_test_file25(old_path1, n_test)
    ux2 = read_test_file25(old_path2, n_test)
    uy2 = read_test_file25(old_path3, n_test)
    uz2 = read_test_file25(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file26(path, num):
    result = np.empty((num, 125000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file26(path, num):
    result = np.empty((num, 125000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff13"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26(old_path1, n_train)
    ux1 = read_train_file26(old_path2, n_train)
    uy1 = read_train_file26(old_path3, n_train)
    uz1 = read_train_file26(old_path4, n_train)
    press2 = read_test_file26(old_path1, n_test)
    ux2 = read_test_file26(old_path2, n_test)
    uy2 = read_test_file26(old_path3, n_test)
    uz2 = read_test_file26(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file27(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file27(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data27(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file27(old_path1, n_train)
    ux1 = read_train_file27(old_path2, n_train)
    uy1 = read_train_file27(old_path3, n_train)
    uz1 = read_train_file27(old_path4, n_train)
    press2 = read_test_file27(old_path1, n_test)
    ux2 = read_test_file27(old_path2, n_test)
    uy2 = read_test_file27(old_path3, n_test)
    uz2 = read_test_file27(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file28(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file28(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data28(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file28(old_path1, n_train)
    ux1 = read_train_file28(old_path2, n_train)
    uy1 = read_train_file28(old_path3, n_train)
    uz1 = read_train_file28(old_path4, n_train)
    press2 = read_test_file28(old_path1, n_test)
    ux2 = read_test_file28(old_path2, n_test)
    uy2 = read_test_file28(old_path3, n_test)
    uz2 = read_test_file28(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file28_addition(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
        result[i, :] = res
    return result


def read_test_file28_addition(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[49:149, :, 49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data28_addition(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file28_addition(old_path1, n_train)
    ux1 = read_train_file28_addition(old_path2, n_train)
    uy1 = read_train_file28_addition(old_path3, n_train)
    uz1 = read_train_file28_addition(old_path4, n_train)
    press2 = read_test_file28_addition(old_path1, n_test)
    ux2 = read_test_file28_addition(old_path2, n_test)
    uy2 = read_test_file28_addition(old_path3, n_test)
    uz2 = read_test_file28_addition(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file29(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[49:149,:,49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file29(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[49:149,:,49:149]
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data29(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file29(old_path1, n_train)
    ux1 = read_train_file29(old_path2, n_train)
    uy1 = read_train_file29(old_path3, n_train)
    uz1 = read_train_file29(old_path4, n_train)
    press2 = read_test_file29(old_path1, n_test)
    ux2 = read_test_file29(old_path2, n_test)
    uy2 = read_test_file29(old_path3, n_test)
    uz2 = read_test_file29(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file30(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        file = file[49:149,:,49:149]
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file30(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        file = file[49:149,:,49:149]
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data30(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file30(old_path1, n_train)
    ux1 = read_train_file30(old_path2, n_train)
    uy1 = read_train_file30(old_path3, n_train)
    uz1 = read_train_file30(old_path4, n_train)
    press2 = read_test_file30(old_path1, n_test)
    ux2 = read_test_file30(old_path2, n_test)
    uy2 = read_test_file30(old_path3, n_test)
    uz2 = read_test_file30(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file31(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file31(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data31(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff16"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file31(old_path1, n_train)
    ux1 = read_train_file31(old_path2, n_train)
    uy1 = read_train_file31(old_path3, n_train)
    uz1 = read_train_file31(old_path4, n_train)
    press2 = read_test_file31(old_path1, n_test)
    ux2 = read_test_file31(old_path2, n_test)
    uy2 = read_test_file31(old_path3, n_test)
    uz2 = read_test_file31(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file32(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file32(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data32(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff16"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file32(old_path1, n_train)
    ux1 = read_train_file32(old_path2, n_train)
    uy1 = read_train_file32(old_path3, n_train)
    uz1 = read_train_file32(old_path4, n_train)
    press2 = read_test_file32(old_path1, n_test)
    ux2 = read_test_file32(old_path2, n_test)
    uy2 = read_test_file32(old_path3, n_test)
    uz2 = read_test_file32(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file33(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for i in range(200):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file33(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = np.empty(2000000, dtype=np.float32)
        for i in range(200):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data33(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff17"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file33(old_path1, n_train)
    ux1 = read_train_file33(old_path2, n_train)
    uy1 = read_train_file33(old_path3, n_train)
    uz1 = read_train_file33(old_path4, n_train)
    press2 = read_test_file33(old_path1, n_test)
    ux2 = read_test_file33(old_path2, n_test)
    uy2 = read_test_file33(old_path3, n_test)
    uz2 = read_test_file33(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file34(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file34(path, num):
    result = np.empty((num, 2000000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data34(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff17"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file34(old_path1, n_train)
    ux1 = read_train_file34(old_path2, n_train)
    uy1 = read_train_file34(old_path3, n_train)
    uz1 = read_train_file34(old_path4, n_train)
    press2 = read_test_file34(old_path1, n_test)
    ux2 = read_test_file34(old_path2, n_test)
    uy2 = read_test_file34(old_path3, n_test)
    uz2 = read_test_file34(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file35(path, num, begin):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+begin}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_test_file35(path, num, begin):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+begin}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for i in range(50):
            res[i * 10000: (i + 1) * 10000] = file[:, :, i].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data35(n_train, n_test, begin_train, begin_test):
    root = "../../../../mnt/data3/xzx/suboff16"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file35(old_path1, n_train, begin_train)
    ux1 = read_train_file35(old_path2, n_train, begin_train)
    uy1 = read_train_file35(old_path3, n_train, begin_train)
    uz1 = read_train_file35(old_path4, n_train, begin_train)
    press2 = read_test_file35(old_path1, n_test, begin_test)
    ux2 = read_test_file35(old_path2, n_test, begin_test)
    uy2 = read_test_file35(old_path3, n_test, begin_test)
    uz2 = read_test_file35(old_path4, n_test, begin_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file36(path, num, begin):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+begin}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file36(path, num, begin):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+begin}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data36(n_train, n_test, begin_train, begin_test):

    root = "../../../../mnt/data3/xzx/suboff16"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file36(old_path1, n_train, begin_train)
    ux1 = read_train_file36(old_path2, n_train, begin_train)
    uy1 = read_train_file36(old_path3, n_train, begin_train)
    uz1 = read_train_file36(old_path4, n_train, begin_train)
    press2 = read_test_file36(old_path1, n_test, begin_test)
    ux2 = read_test_file36(old_path2, n_test, begin_test)
    uy2 = read_test_file36(old_path3, n_test, begin_test)
    uz2 = read_test_file36(old_path4, n_test, begin_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file49(path, num):
    result = np.empty((num, 5000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        result[i, :] = file[49:149,:,100].flatten()
    return result


def read_test_file49(path, num):
    result = np.empty((num, 5000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1250}.npy"
        file = np.load(filename)
        result[i, :] = file[49:149, :, 100].flatten()
    return result


def read_multi_re_cylinder_data49(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file49(old_path1, n_train)
    ux1 = read_train_file49(old_path2, n_train)
    uy1 = read_train_file49(old_path3, n_train)
    uz1 = read_train_file49(old_path4, n_train) # 1250 5000
    t1 = np.repeat(np.arange(n_train)[:, np.newaxis], 5000, axis=1)
    press2 = read_test_file49(old_path1, n_test)
    ux2 = read_test_file49(old_path2, n_test)
    uy2 = read_test_file49(old_path3, n_test)
    uz2 = read_test_file49(old_path4, n_test)
    t2 = np.repeat(np.arange(n_train, n_train+n_test)[:, np.newaxis], 5000, axis=1)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    t1 = torch.as_tensor(t1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)
    t2 = torch.as_tensor(t2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1, t1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2, t2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2, t1, t2

    return data_train, data_test
def read_train_file26_1(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file26_1(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_1(n_train, n_test):
    root = "../../../../mnt/data3/xzx/re200"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_1(old_path1, n_train)
    ux1 = read_train_file26_1(old_path2, n_train)
    uy1 = read_train_file26_1(old_path3, n_train)
    uz1 = read_train_file26_1(old_path4, n_train)
    press2 = read_test_file26_1(old_path1, n_test)
    ux2 = read_test_file26_1(old_path2, n_test)
    uy2 = read_test_file26_1(old_path3, n_test)
    uz2 = read_test_file26_1(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file26_2(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file26_2(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_2(n_train, n_test):

    root = "../../../../mnt/data3/xzx/re200"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_2(old_path1, n_train)
    ux1 = read_train_file26_2(old_path2, n_train)
    uy1 = read_train_file26_2(old_path3, n_train)
    uz1 = read_train_file26_2(old_path4, n_train)
    press2 = read_test_file26_2(old_path1, n_test)
    ux2 = read_test_file26_2(old_path2, n_test)
    uy2 = read_test_file26_2(old_path3, n_test)
    uz2 = read_test_file26_2(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test
def read_train_file26_3(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file26_3(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_3(n_train, n_test):
    root = "../../../../mnt/data3/xzx/re300"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_3(old_path1, n_train)
    ux1 = read_train_file26_3(old_path2, n_train)
    uy1 = read_train_file26_3(old_path3, n_train)
    uz1 = read_train_file26_3(old_path4, n_train)
    press2 = read_test_file26_3(old_path1, n_test)
    ux2 = read_test_file26_3(old_path2, n_test)
    uy2 = read_test_file26_3(old_path3, n_test)
    uz2 = read_test_file26_3(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file26_4(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file26_4(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_4(n_train, n_test):

    root = "../../../../mnt/data3/xzx/re300"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_4(old_path1, n_train)
    ux1 = read_train_file26_4(old_path2, n_train)
    uy1 = read_train_file26_4(old_path3, n_train)
    uz1 = read_train_file26_4(old_path4, n_train)
    press2 = read_test_file26_4(old_path1, n_test)
    ux2 = read_test_file26_4(old_path2, n_test)
    uy2 = read_test_file26_4(old_path3, n_test)
    uz2 = read_test_file26_4(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file26_5(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file26_5(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_5(n_train, n_test):
    root = "../../../../mnt/data3/xzx/re250"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_5(old_path1, n_train)
    ux1 = read_train_file26_5(old_path2, n_train)
    uy1 = read_train_file26_5(old_path3, n_train)
    uz1 = read_train_file26_5(old_path4, n_train)
    press2 = read_test_file26_5(old_path1, n_test)
    ux2 = read_test_file26_5(old_path2, n_test)
    uy2 = read_test_file26_5(old_path3, n_test)
    uz2 = read_test_file26_5(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file26_6(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+4500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file26_6(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+5500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data26_6(n_train, n_test):

    root = "../../../../mnt/data3/xzx/re250"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file26_6(old_path1, n_train)
    ux1 = read_train_file26_6(old_path2, n_train)
    uy1 = read_train_file26_6(old_path3, n_train)
    uz1 = read_train_file26_6(old_path4, n_train)
    press2 = read_test_file26_6(old_path1, n_test)
    ux2 = read_test_file26_6(old_path2, n_test)
    uy2 = read_test_file26_6(old_path3, n_test)
    uz2 = read_test_file26_6(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file65(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file65(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data65(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff33"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file65(old_path1, n_train)
    ux1 = read_train_file65(old_path2, n_train)
    uy1 = read_train_file65(old_path3, n_train)
    uz1 = read_train_file65(old_path4, n_train)
    press2 = read_test_file65(old_path1, n_test)
    ux2 = read_test_file65(old_path2, n_test)
    uy2 = read_test_file65(old_path3, n_test)
    uz2 = read_test_file65(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file66(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file66(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+1000}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data66(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff33"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file66(old_path1, n_train)
    ux1 = read_train_file66(old_path2, n_train)
    uy1 = read_train_file66(old_path3, n_train)
    uz1 = read_train_file66(old_path4, n_train)
    press2 = read_test_file66(old_path1, n_test)
    ux2 = read_test_file66(old_path2, n_test)
    uy2 = read_test_file66(old_path3, n_test)
    uz2 = read_test_file66(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file69(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_test_file69(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500+1250}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(100):
            res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data69(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff33"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file69(old_path1, n_train)
    ux1 = read_train_file69(old_path2, n_train)
    uy1 = read_train_file69(old_path3, n_train)
    uz1 = read_train_file69(old_path4, n_train)
    press2 = read_test_file69(old_path1, n_test)
    ux2 = read_test_file69(old_path2, n_test)
    uy2 = read_test_file69(old_path3, n_test)
    uz2 = read_test_file69(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file70(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_test_file70(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500+1250}.npy"
        file = np.load(filename)
        res = file.flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data70(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff33"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file70(old_path1, n_train)
    ux1 = read_train_file70(old_path2, n_train)
    uy1 = read_train_file70(old_path3, n_train)
    uz1 = read_train_file70(old_path4, n_train)
    press2 = read_test_file70(old_path1, n_test)
    ux2 = read_test_file70(old_path2, n_test)
    uy2 = read_test_file70(old_path3, n_test)
    uz2 = read_test_file70(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file70_addition(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
        result[i, :] = res
    return result


def read_test_file70_addition(path, num):
    result = np.empty((num, 500000), dtype=np.float32)
    for i in range(num):
        filename = path + f"/{i+33500+1250}.npy"
        file = np.load(filename)
        res = np.empty(500000, dtype=np.float32)
        for j in range(50):
            res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
        result[i, :] = res
    return result


def read_multi_re_cylinder_data70_addition(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff33"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file70_addition(old_path1, n_train)
    ux1 = read_train_file70_addition(old_path2, n_train)
    uy1 = read_train_file70_addition(old_path3, n_train)
    uz1 = read_train_file70_addition(old_path4, n_train)
    press2 = read_test_file70_addition(old_path1, n_test)
    ux2 = read_test_file70_addition(old_path2, n_test)
    uy2 = read_test_file70_addition(old_path3, n_test)
    uz2 = read_test_file70_addition(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test
def read_train_file73(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = np.empty(500000, dtype=np.float32)
    for j in range(100):
        res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
    result[0, :] = res
    return result


def read_test_file73(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = np.empty(500000, dtype=np.float32)
    for j in range(100):
        res[j * 5000: (j + 1) * 5000] = file[:, :, j].flatten()
    result[0, :] = res
    return result


def read_multi_re_cylinder_data73(n_train, n_test):
    root = "./suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file73(old_path1, n_train)
    ux1 = read_train_file73(old_path2, n_train)
    uy1 = read_train_file73(old_path3, n_train)
    uz1 = read_train_file73(old_path4, n_train)
    press2 = read_test_file73(old_path1, n_test)
    ux2 = read_test_file73(old_path2, n_test)
    uy2 = read_test_file73(old_path3, n_test)
    uz2 = read_test_file73(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file74(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = file.flatten()
    result[0, :] = res

    return result


def read_test_file74(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = file.flatten()
    result[0, :] = res

    return result


def read_multi_re_cylinder_data74(n_train, n_test):

    root = "./suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file74(old_path1, n_train)
    ux1 = read_train_file74(old_path2, n_train)
    uy1 = read_train_file74(old_path3, n_train)
    uz1 = read_train_file74(old_path4, n_train)
    press2 = read_test_file74(old_path1, n_test)
    ux2 = read_test_file74(old_path2, n_test)
    uy2 = read_test_file74(old_path3, n_test)
    uz2 = read_test_file74(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test

def read_train_file74_addition(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = np.empty(500000, dtype=np.float32)
    for j in range(50):
        res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
    result[0, :] = res

    return result


def read_test_file74_addition(path, num):
    result = np.empty((1, 500000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    file = file[49:149, :, 49:149]
    res = np.empty(500000, dtype=np.float32)
    for j in range(50):
        res[j * 10000: (j + 1) * 10000] = file[:, j, :].flatten()
    result[0, :] = res

    return result


def read_multi_re_cylinder_data74_addition(n_train, n_test):
    root = "./suboff8"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file74_addition(old_path1, n_train)
    ux1 = read_train_file74_addition(old_path2, n_train)
    uy1 = read_train_file74_addition(old_path3, n_train)
    uz1 = read_train_file74_addition(old_path4, n_train)
    press2 = read_test_file74_addition(old_path1, n_test)
    ux2 = read_test_file74_addition(old_path2, n_test)
    uy2 = read_test_file74_addition(old_path3, n_test)
    uz2 = read_test_file74_addition(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test
def read_train_file77(path, num):
    result = np.empty((1, 875000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    res = np.empty(875000, dtype=np.float32)
    for j in range(50):
        res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
    result[0, :] = res

    return result


def read_test_file77(path, num):
    result = np.empty((1, 875000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    res = np.empty(875000, dtype=np.float32)
    for j in range(50):
        res[j * 17500: (j + 1) * 17500] = file[:, :, j].flatten()
    result[0, :] = res
    return result


def read_multi_re_cylinder_data77(n_train, n_test):
    root = "../../../../mnt/data3/xzx/suboff6"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file77(old_path1, n_train)
    ux1 = read_train_file77(old_path2, n_train)
    uy1 = read_train_file77(old_path3, n_train)
    uz1 = read_train_file77(old_path4, n_train)
    press2 = read_test_file77(old_path1, n_test)
    ux2 = read_test_file77(old_path2, n_test)
    uy2 = read_test_file77(old_path3, n_test)
    uz2 = read_test_file77(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test


def read_train_file78(path, num):
    result = np.empty((1, 875000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    res = file.flatten()
    result[0, :] = res

    return result


def read_test_file78(path, num):
    result = np.empty((1, 875000), dtype=np.float32)
    filename = path + f"/{num}.npy"
    file = np.load(filename)
    res = file.flatten()
    result[0, :] = res

    return result


def read_multi_re_cylinder_data78(n_train, n_test):

    root = "../../../../mnt/data3/xzx/suboff6"
    old_path1 = root + f"/p"
    old_path2 = root + f"/ux"
    old_path3 = root + f"/uy"
    old_path4 = root + f"/uz"

    press1 = read_train_file78(old_path1, n_train)
    ux1 = read_train_file78(old_path2, n_train)
    uy1 = read_train_file78(old_path3, n_train)
    uz1 = read_train_file78(old_path4, n_train)
    press2 = read_test_file78(old_path1, n_test)
    ux2 = read_test_file78(old_path2, n_test)
    uy2 = read_test_file78(old_path3, n_test)
    uz2 = read_test_file78(old_path4, n_test)

    press1 = torch.as_tensor(press1, dtype=torch.float32).unsqueeze(dim=-1)
    ux1 = torch.as_tensor(ux1, dtype=torch.float32).unsqueeze(dim=-1)
    uy1 = torch.as_tensor(uy1, dtype=torch.float32).unsqueeze(dim=-1)
    uz1 = torch.as_tensor(uz1, dtype=torch.float32).unsqueeze(dim=-1)
    press2 = torch.as_tensor(press2, dtype=torch.float32).unsqueeze(dim=-1)
    ux2 = torch.as_tensor(ux2, dtype=torch.float32).unsqueeze(dim=-1)
    uy2 = torch.as_tensor(uy2, dtype=torch.float32).unsqueeze(dim=-1)
    uz2 = torch.as_tensor(uz2, dtype=torch.float32).unsqueeze(dim=-1)

    data_train = torch.cat((press1, ux1, uy1, uz1), dim=-1)
    data_test = torch.cat((press2, ux2, uy2, uz2), dim=-1)
    del press1, ux1, uy1, uz1, press2, ux2, uy2, uz2

    return data_train, data_test
class CylinderDatasetMultiRe1(Dataset):
    #   2 1000 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y


class CylinderDatasetMultiRe4(Dataset):
    #   2 1000 875000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 875000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 875000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 875000 4
        return x1, x2, y


class CylinderDatasetMultiRe6(Dataset):
    #   2 1000 875000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 875000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 875000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 875000 4
        return x1, x2, y

class CylinderDatasetMultiRe7(Dataset):
    #   2 1250 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y

class CylinderDatasetMultiRe8(Dataset):
    #   2 1250 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y

class CylinderDatasetMultiRe9(Dataset):
    #   2 1250 1000000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 2000000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 2000000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y


class CylinderDatasetMultiRe14(Dataset):
    #   3 1250 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        x3 = self.data[2, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, x3, y

class CylinderDatasetMultiRe25(Dataset):
    #   1 1250 5000 5
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x = self.data[0, idx:idx+self.tw, :, 0:4] # 1 5000 4
        t = self.data[0, idx+self.tw+self.push_forward-1, :, 4:5]    # 5000 1
        y = self.data[0, idx+self.tw+self.push_forward-1, :, 0:4]  # 5000 4
        return x, t, y
class CylinderDatasetMultiRe26(Dataset):
    #   4 1000 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        x3 = self.data[2, idx:idx+self.tw, ...] # 1 500000 4
        x4 = self.data[3, idx:idx+self.tw, ...] # 1 500000 4
        y1 = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        y2 = self.data[3, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, x3, x4, y1, y2


class CylinderDatasetMultiRe28(Dataset):
    #   2 1250 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y, torch.as_tensor(idx).unsqueeze(-1)

class CylinderDatasetMultiRe30(Dataset):
    #   2 1250 500000 4
    def __init__(self, data, tw=1, push_forward=1):
        super().__init__()
        self.data = data
        self.tw = tw
        self.push_forward = push_forward
        self.length_one_block = self.data.shape[1]

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x1 = self.data[0, idx:idx+self.tw, ...] # 1 500000 4
        x2 = self.data[1, idx:idx+self.tw, ...] # 1 500000 4
        y = self.data[1, idx+self.tw+self.push_forward-1, ...] # 500000 4
        return x1, x2, y, torch.as_tensor(idx/1249).unsqueeze(-1)
