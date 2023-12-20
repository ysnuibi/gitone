import numpy as np
import pandas as pd

# 用DataFrame来储存数据，格式为userid, itemid, rating
df = pd.read_csv('./data.txt', sep='\t', header=None)
df.drop(3, inplace=True, axis=1) # 去掉时间戳
df.columns = ['uid', 'iid', 'rating']
# print(df.columns)
# print(df)

# 随机打乱划分训练和测试集
df = df.sample(frac=1, random_state=0)
train_set = df.iloc[:int(len(df)*0.75)]
test_set = df.iloc[int(len(df)*0.75):]

n_users = max(df.uid)+1 # +1是因为uid最小是从1开始，懒得在pu切片中-1，所以多一维空着也没事
n_items = max(df.iid)+1


class SVDpp(object):
    def __init__(self, n_epochs, n_users, n_items, n_factors, lr, reg_rate, random_seed=0):
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_rate = reg_rate
        self.n_factors = n_factors
        np.random.seed(random_seed)
        self.pu = np.random.randn(n_users, n_factors) / np.sqrt(n_factors)
        self.qi = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)
        self.yj = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)
        self.bu = np.zeros(n_users, np.double)
        self.bi = np.zeros(n_items, np.double)
        self.global_bias = 0
        self.Iu = dict()

    def reg_sum_yj(self, u, i):
        sum_yj = np.zeros(self.n_factors, np.double)
        for j in self.Iu[u]:
            sum_yj += self.yj[j]
        return sum_yj / np.sqrt(len(self.Iu[u]))

    def predict(self, u, i, feedback_vec_reg):
        return self.global_bias + self.bu[u] + self.bi[i] + np.dot(self.qi[i], self.pu[u] + feedback_vec_reg)

    def fit(self, train_set, verbose=True):
        self.global_bias = np.mean(train_set.rating)
        # 将用户打过分的记录到Iu字典中，key为uid，value为打过分的iid的list
        g = train_set.groupby(['uid'])
        for uid, df_uid in g:
            self.Iu[uid] = list(df_uid.iid)

        for epoch in range(self.n_epochs):
            square_err = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.iid, row.rating
                feedback_vec_reg = self.reg_sum_yj(u, i)
                error = r - self.predict(u, i, feedback_vec_reg)
                square_err += error ** 2
                self.bu[u] += self.lr * (error - self.reg_rate * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg_rate * self.bi[i])
                tmp_pu = self.pu[u]
                tmp_qi = self.qi[i]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * (tmp_pu + feedback_vec_reg) - self.reg_rate * self.qi[i])
                for j in self.Iu[u]:
                    self.yj[j] += self.lr * (error / np.sqrt(len(self.Iu[u])) * tmp_qi - self.reg_rate * self.yj[j])
            if verbose == True:
                rmse = np.sqrt(square_err / len(train_set))
                print('epoch: %d, rmse: %.4f' % (epoch, rmse))
        return self

    def fit_MAE(self, train_set, verbose=True):
        self.global_bias = np.mean(train_set.rating)
        # 将用户打过分的记录到Iu字典中，key为uid，value为打过分的iid的list
        g = train_set.groupby(['uid'])
        for uid, df_uid in g:
            self.Iu[uid] = list(df_uid.iid)
        for epoch in range(self.n_epochs):
            square_err = 0
            for index, row in train_set.iterrows():
                u, i, r = row.uid, row.iid, row.rating
                feedback_vec_reg = self.reg_sum_yj(u, i)
                error = r - self.predict(u, i, feedback_vec_reg)
                error1 = abs(error)
                square_err += error1
                # mse += error ** 2
                self.bu[u] += self.lr * (error - self.reg_rate * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg_rate * self.bi[i])
                tmp = self.pu[u]
                tmp_qi = self.qi[i]
                self.pu[u] += self.lr * (error * self.qi[i] - self.reg_rate * self.pu[u])
                self.qi[i] += self.lr * (error * tmp - self.reg_rate * self.qi[i])
            if verbose == True:
                MAE = abs(square_err) / len(train_set)
                print('epoch: %d, MAE: %.4f' % (epoch, MAE))



        return self

    def test(self, test_set):
        predictions = test_set.apply(lambda x: self.predict(x.uid, x.iid, self.reg_sum_yj(x.uid, x.iid)), axis=1)
        rmse = np.sqrt(np.sum((test_set.rating - predictions) ** 2) / len(test_set))
        return rmse


svdpp = SVDpp(n_epochs=100, n_users=n_users, n_items=n_items, n_factors=45, lr=0.01, reg_rate=0.3)
svdpp.fit(train_set, verbose=True)
#svdpp.fit_MAE(train_set, verbose=True)
svdpp.test(test_set)
# 0.9510302683304096
svdpp.predict(120, 282, svdpp.reg_sum_yj(120, 282))
# 真实评分为4，预测为3.5370712737668204

