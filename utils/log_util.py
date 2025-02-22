# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:51
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import inspect


def cy_log(*args, positionable=False):
    if positionable:
        # 打印定位信息
        file_path = inspect.currentframe().f_back.f_code.co_filename
        lineno = inspect.currentframe().f_back.f_lineno  # f_back只追溯一层,到调用cy_log的地方
        print(f"{file_path}:{lineno}:")
    print('cfushn >>> ', *args)  # 这里必须使用*解包, 否则args会以列表形式输出


if __name__ == "__main__":
    cy_log("ssdf", 123)
    cy_log("ssdf", 123, positionable=True)


    def lll(*args, positionable=False):
        return cy_log(*args, positionable=positionable)


    lll("ssdf", 123)
    lll("/home/cy/workdir/cfushn-ccgan_0/flow/train_ccgan.py:88 ", 1223, positionable=True)
    print("/home/cy/workdir/cfushn-ccgan_0/flow/train_ccgan.py:88")
    print("\"/home/cy/workdir/cfushn-ccgan_0/flow/train_ccgan.py:88\"")
    print("cfushn.=dfg$ /home/cy/workdir/cfushn-ccgan_0/flow/train_ccgan.py:88")
