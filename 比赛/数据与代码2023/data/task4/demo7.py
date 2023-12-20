import math

def romberg_integration(func, a, b, n):
    """
    使用Romberg算法计算定积分的近似值。
    :param func: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param n: 迭代次数
    :return: 定积分的近似值
    """
    if n == 0:
        return (b - a) * (func(a) + func(b)) / 2
    R = romberg_integration(func, a, b, n - 1)
    h = b - a
    Rn = h * (func(a) + func(b)) / 2 + R * h**2 / (4 ** n)
    for i in range(1, n + 1):
        Rn += R * h**i / (2 ** i)
        R = Rn - R * h**i / (2 ** i)
    return Rn
# 示例：计算函数f(x) = x^2在区间[0, 1]上的定积分近似值，迭代次数为5
def f(x):
    return x**2
a = 0
b = 1
n = 5
result = romberg_integration(f, a, b, n)
print("定积分的近似值为：", result)