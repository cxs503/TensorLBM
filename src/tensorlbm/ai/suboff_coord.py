import numpy as np


def coord_ori1():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori2():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords


def coord_ori4():
    x_coords = np.linspace(-5, 15, 400)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 5, 100)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori7():
    x_coords = np.linspace(-10, 30, 800)[99:799:2]  # 列方向（x）
    y_coords = np.linspace(-5, 5, 200)[49:149:2]  # 行方向（y）
    z_coords = np.linspace(0, 10, 200)[49:149:2]  # z方向
    blocks = np.empty((875000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*17500:(i+1)*17500, :] = coords
    return blocks


def coord_ori8():
    x_coords = np.linspace(-10, 30, 800)[99:799:2]  # 列方向（x）
    y_coords = np.linspace(-5, 5, 200)[49:149:2]  # 行方向（y）
    z_coords = np.linspace(0, 10, 200)[49:149:2]  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords


def coord_ori11():
    x_coords = np.linspace(-10, 30, 800)[99:799:2]  # 列方向（x）
    y_coords = np.linspace(-5, 5, 200)[49:149:2]  # 行方向（y）
    z_coords = np.linspace(0, 10, 200)[49:149:2]  # z方向
    blocks = np.empty((875000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*17500:(i+1)*17500, :] = coords
    return blocks


def coord_ori12():
    x_coords = np.linspace(-10, 30, 800)[99:799:2]  # 列方向（x）
    y_coords = np.linspace(-5, 5, 200)[49:149:2]  # 行方向（y）
    z_coords = np.linspace(0, 10, 200)[49:149:2]  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori13():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori14():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori15():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-10, 10, 200)  # 行方向（y）
    z_coords = np.linspace(0, 20, 200)  # z方向
    blocks = np.empty((2000000, 3), dtype=np.float32)
    for i in range(200):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori16():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-10, 10, 200)  # 行方向（y）
    z_coords = np.linspace(0, 20, 200)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords


def coord_ori17():
    x_coords = np.linspace(-1.25, 1.25, 25)  # 列方向（x）
    y_coords = np.linspace(-10, 10, 200)  # 行方向（y）
    z_coords = np.linspace(0, 20, 200)  # z方向
    blocks = np.empty((1000000, 3), dtype=np.float32)
    for i in range(200):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*5000:(i+1)*5000, :] = coords
    return blocks


def coord_ori18():
    x_coords = np.linspace(-1.25, 1.25, 25)  # 列方向（x）
    y_coords = np.linspace(-10, 10, 200)  # 行方向（y）
    z_coords = np.linspace(0, 20, 200)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori25():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((2000000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*2500:(i+1)*2500, :] = coords
    return blocks


def coord_ori26():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori27():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(100):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*5000:(i+1)*5000, :] = coords
    return blocks


def coord_ori28():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori28_addition():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(x_coords[i])
        y_vals = y_coords
        z_vals = z_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [0, 1, 2]]
        blocks[i * 10000:(i + 1) * 10000, :] = coords
    return blocks

def coord_ori29():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(100):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*5000:(i+1)*5000, :] = coords
    return blocks


def coord_ori30():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori31():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(2.5, -2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori32():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(2.5, -2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori49():
    x_coords = np.linspace(-1.25, 3.75, 50)  # 列方向（x）
    y_coords = np.linspace(-5, 5, 100)  # 行方向（y）
    z_coords = np.linspace(0, 10, 100)  # z方向
    x_vals = np.array(z_coords[50])
    y_vals = y_coords
    z_vals = x_coords
    n = 1
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [2, 1, 0]]
    coords = coords[:, 0:2]
    return coords

def coord_ori51():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(2.5, -2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori52():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(2.5, -2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords

def coord_ori65():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    blocks = np.empty((500000, 3), dtype=np.float32)
    for i in range(50):
        x_vals = np.array(z_coords[i])
        y_vals = y_coords
        z_vals = x_coords
        n = 1
        m = len(y_vals)
        k = len(z_vals)
        # 生成各轴的扩展数组
        x = np.repeat(x_vals, m * k)
        y = np.tile(np.repeat(y_vals, k), n)
        z = np.tile(z_vals, n * m)
        # 合并成(N, 3)数组
        coords = np.stack([x, y, z], axis=1)
        coords = coords[:, [2, 1, 0]]
        blocks[i*10000:(i+1)*10000, :] = coords
    return blocks


def coord_ori66():
    x_coords = np.linspace(-5, 15, 200)  # 列方向（x）
    y_coords = np.linspace(-2.5, 2.5, 50)  # 行方向（y）
    z_coords = np.linspace(0, 5, 50)  # z方向
    x_vals = y_coords
    y_vals = x_coords
    z_vals = z_coords
    n = len(x_vals)
    m = len(y_vals)
    k = len(z_vals)
    # 生成各轴的扩展数组
    x = np.repeat(x_vals, m * k)
    y = np.tile(np.repeat(y_vals, k), n)
    z = np.tile(z_vals, n * m)
    # 合并成(N, 3)数组
    coords = np.stack([x, y, z], axis=1)
    coords = coords[:, [1, 0, 2]]
    return coords
