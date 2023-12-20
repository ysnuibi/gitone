import numpy as np
import pandas as pd

# 用DataFrame来储存数据，格式为userid, itemid, rating
df = pd.read_csv('ratings.dat', sep='::', header=None,engine='python')
df.drop(3, inplace=True, axis=1)  # 去掉时间戳
df.columns = ['uid', 'iid', 'rating']

# 随机打乱划分训练和测试集
df = df.sample(frac=1, random_state=0)
train_set = df.iloc[:int(len(df) * 0.75)]
test_set = df.iloc[int(len(df) * 0.75):]

n_users = max(df.uid) + 1  # +1是因为uid最小是从1开始，懒得在pu切片中-1，所以多一维空着也没事
n_items = max(df.iid) + 1


class Funk_SVD(object):
    def __init__(self, n_epochs, n_users, n_items, n_factors, lr, reg_rate, random_seed=0):
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_rate = reg_rate
        np.random.seed(random_seed)
        self.pu = np.random.randn(n_users, n_factors) / np.sqrt(n_factors)  # 参数初始化不能太大
        self.qi = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)

    def predict(self, u, i):
        return np.dot(self.qi[i], self.pu[u])

    def fit(self, train_set, verbose=True):
        for epoch in range(self.n_epochs):
            mse = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.iid, row.rating
                error = r - self.predict(u, i)
                mse += error ** 2
                tmp = self.pu[u]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * tmp - self.reg_rate * self.qi[i])
            if verbose == True:
                rmse = np.sqrt(mse / len(train_set))
                print('epoch: %d, rmse: %.4f' % (epoch, rmse))
        return self

    def fit_MAE(self, train_set, verbose=True):
        for epoch in range(self.n_epochs):
            mae = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.iid, row.rating
                error = r - self.predict(u, i)
                error1 = abs(error)
               # mse += error ** 2
                mae += error1
                tmp = self.pu[u]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * tmp - self.reg_rate * self.qi[i])
            if verbose == True:
                MAE = abs(mae) / len(train_set)
                print('epoch: %d, MAE: %.4f' % (epoch, MAE))
        return self

    def fit_1(self, train_set, verbose=True):
        for epoch in range(self.n_epochs):
            mse = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.iid, row.rating
                error = r - self.predict(u, i)
                mse += error ** 2
                tmp = self.pu[u]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * tmp - self.reg_rate * self.qi[i])
            if verbose == True:
                mse = mse / len(train_set)
                print('epoch: %d, mse: %.4f' % (epoch, mse))
        return self

    def test(self, test_set):
        predictions = test_set.apply(lambda x: self.predict(x.uid, x.iid), axis=1)
        rmse = np.sqrt(np.sum((test_set.rating - predictions) ** 2) / len(test_set))
       # mse = (np.sum((test_set.rating - predictions) ** 2) / len(test_set))
        return rmse
    #
    # def test_1(self, test_set):
    #     predictions = test_set.apply(lambda x: self.predict(x.uid, x.iid), axis=1)
    #     #rmse = np.sqrt(np.sum((test_set.rating - predictions) ** 2) / len(test_set))
    #     mse = (np.sum((test_set.rating - predictions) ** 2) / len(test_set))
    #     return mse

    # def test_MAE(self, test_set):
    #     predictions = test_set.apply(lambda x: self.predict(x.uid, x.iid), axis=1)
    #     #rmse = np.sqrt(np.sum((test_set.rating - predictions) ** 2) / len(test_set))
    #     #mse = (np.sum((test_set.rating - predictions) ** 2) / len(test_set))
    #     MAE = (test_set.rating - predictions) / float(len(test_set))
    #     return MAE



funk_svd = Funk_SVD(n_epochs=100, n_users=n_users, n_items=n_items, n_factors=5, lr=0.01, reg_rate=0.03)
funk_svd.fit(train_set, verbose=True)
# funk_svd.fit_1(train_set, verbose=True)
funk_svd.fit_MAE(train_set, verbose=True)
funk_svd.test(test_set)
#funk_svd.test_MAE(test_set)
#print(funk_svd.test(test_set))
# 0.9872467462373891
funk_svd.predict(120, 282)
#print(funk_svd.predict(123, 100))
# 测试集中的某一条数据，真实评分为4，预测为3.233268069895416

