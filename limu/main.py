def softmax(x):
    X_exp = torch.exp(x)
    partition = X_exp.sum(1,keepdim = True)
    return X_exp / partition