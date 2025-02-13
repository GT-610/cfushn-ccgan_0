# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:51
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

def cy_log(*args):
    print('cfushn >>> ', *args)  # 这里必须使用*解包, 否则args会以列表形式输出
