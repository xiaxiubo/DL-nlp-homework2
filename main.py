import numpy as np


def data_generate(theta0):
    s1, s2, p, q, r = theta0
    n_max = 10000
    data = []
    for i in range(n_max):
        t1 = np.random.rand()
        t2 = np.random.rand()
        if t1 < s1:  # 仍第一个硬币
            data.append(1) if t2 < p else data.append(0)  # 仍出正面加个1,背面加个0
        elif t1 < s1 + s2:  # 仍第二个硬币
            data.append(1) if t2 < q else data.append(0)
        else:  # 仍第三个硬币
            data.append(1) if t2 < r else data.append(0)

    return data


# 通过xi的分布计算出miu1，miu2
def pd_xi(xi, theta):
    s1, s2, p, q, r = theta
    pd = p**xi*(1-p)**(1-xi)*s1 + q**xi*(1-q)**(1-xi)*s2 + r**xi*(1-r)**(1-xi)*(1-s1-s2)
    miu1 = p**xi*(1-p)**(1-xi)*s1/pd
    miu2 = q**xi*(1-q)**(1-xi)*s2/pd
    return miu1, miu2


# EM算法的E步,通过参数theta的值计算出各xi的miu1,miu2
def e_bu(data, theta):
    u1 = []
    u2 = []
    for d in data:
        miu1, miu2 = pd_xi(d, theta)
        u1.append(miu1)
        u2.append(miu2)
    return u1, u2


# EM算法的M步, 通过E步计算出的u1,u2来推测最大概率对应的theta值.
def m_bu(data, u1, u2):
    temp1 = 0
    temp2 = 0
    temp3 = 0
    s1 = sum(u1)/len(u1)
    s2 = sum(u2)/len(u2)
    for d, miu1, miu2 in zip(data, u1, u2):
        temp1 += miu1*d
        temp2 += miu2*d
        temp3 += (1-miu1-miu2)*d
    p = temp1/sum(u1)
    q = temp2/sum(u2)
    r = temp3/(len(data)-sum(u1)-sum(u2))
    theta = [s1, s2, p, q, r]
    return theta


# 运行，最大迭代次数为n_max，超过该次数系统自动停止
def run(data, theta, n_max):
    for i in range(n_max):
        u1, u2 = e_bu(data, theta)
        theta = m_bu(data, u1, u2)
        print('迭代第', i+1, '次参数结果为: ', [float(format(j, '.3f')) for j in theta])
    return theta

if __name__ == '__main__':
    # 所设定的参数theta =  [s1 s2 p q r]
    theta0 = [0.1, 0.6, 0.4, 0.1, 0.6]
    # 初始化参数，应该是随机的，这里为了测试方便手动输入了，也可以随机,下面四行就是随机生成初始化参数的
    # theta_int = [0.3, 0.4, 0.4, 0.2, 0.9]
    theta_int = np.random.rand(5)
    theta_rand3 = np.random.rand(3)
    theta_rand3 = theta_rand3/sum(theta_rand3)
    theta_int[:2] = theta_rand3[:2]

    # 生成初始化数据：01001001110101111……
    data = data_generate(theta0)
    # 进行EM迭代，并返回最终的参数值th = [s1,s2,p,q,r]
    th = run(data, theta_int, 30)
    print('')
    print('————'*4, '结果如下', '————'*6)
    print('初始参数真值为：', theta0)
    print('由初始参数真值，模型中出现1的概率为：{:.4f}'.format(theta0[0] * theta0[2] + theta0[1] * theta0[3] +
                                             (1 - theta0[0] - theta0[1]) * theta0[4]))
    print('由上述参数，模型随机生成的data中，1和0的比率为:{:.4f}'.format(sum(data) / len(data)))

    print('迭代后的参数为：', [float(format(j, '.3f')) for j in th])
    print('由迭代后的参数，模型随机生成的data中，1和0的比率为:{:.4f}'.format(th[0]*th[2]+th[1]*th[3]+(1-th[0]-th[1])*th[4]))
    print('————'*13)
