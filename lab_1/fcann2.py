import numpy as np

def fcann2_train(X, Y_, iterations=100, param_delta=1):
    Y_ = Y_.astype(int)
    for i in range(iterations):
        N, D = X.shape
        K = max(Y_) + 1

        W = 0.01 * np.random.randn(D, K)
        b = np.zeros((1, K))

        # eksponencirani klasifikacijski rezultati
        scores = np.dot(X, W) + b    # N x C
        expscores = np.exp(scores) # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = expscores / sumexp     # N x C
        logprobs = -np.log(probs[:, Y_])  # N x C

        # gubitak
        loss  = np.sum(logprobs) / N     # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po rezultatu
        dL_ds = probs     # N x C
        dL_ds[:, Y_] -= 1
        dL_ds /= N

        # gradijenti parametara
        grad_W = np.dot(X.T, dL_ds)    # C x D (ili D x C)
        grad_b = np.sum(dL_ds, axis=0, keepdims=True)    # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    scores = np.dot(X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Y_)))
    return W, b

def fcann2_classify(X):
    pass
