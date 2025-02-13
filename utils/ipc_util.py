# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:50
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import signal

# 两个自定义信号量, 供用户处理事件, 通过SIGUSR1 和 SIGUSR2 信号操作其值
__s1 = 0
__s2 = 0


def register_signal_handler():
    """
    注册用户自定义signal_handler,在接收信号时触发触发对应信号量的自定义操作
    """
    def _signal_handler(signum, frame):
        global __s1, __s2
        if signum == signal.SIGUSR1:
            __s1 = 1 - __s1  # 在0,1之间switch
            print(f"================= Received signal {signum}, set v1 = {__s1} ==================")
        elif signum == signal.SIGUSR2:
            __s2 = 1 - __s2  # 在0,1之间switch
            print(f"================= Received signal {signum}, set v2 = {__s2} ==================")

    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGUSR2, _signal_handler)


def get_s1():
    return __s1


def get_s2():
    return __s2
