import numpy as np

def fcann2_train(X, Y_, iterations=1000, param_delta=1e-4, h=5):
    Y_ = Y_.astype(int)
    N, D = X.shape
    K = max(Y_) + 1

    W1 = np.random.randn(D, h)
    b1 = np.zeros((1, h))
    W2 = np.random.randn(h, K)
    b2 = np.zeros((1, K))

    print(W1, b1, W2, b2)

    for i in range(iterations):
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        # eksponencirani klasifikacijski rezultati
        scores = np.dot(hidden_layer, W2) + b2
        expscores = np.exp(scores)

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)

        # logaritmirane vjerojatnosti razreda
        probs = expscores / sumexp
        logprobs = -np.log(probs[range(N), Y_])

        # gubitak
        loss  = np.sum(logprobs) / N

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po rezultatu
        dL_ds = probs
        dL_ds[range(N), Y_] -= 1
        dL_ds /= N

        # gradijenti parametara
        grad_W2 = np.dot(hidden_layer.T, dL_ds)
        grad_b2 = np.sum(dL_ds, axis=0, keepdims=True)

        grad_hidden = np.dot(dL_ds, W2.T)
        grad_hidden[hidden_layer <= 0] = 0

        grad_W1 = np.dot(X.T, grad_hidden)    # C x D (ili D x C)
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)    # C x 1 (ili 1 x C)

        # poboljšani parametri
        W1 += -param_delta * grad_W1
        b1 += -param_delta * grad_b1
        W2 += -param_delta * grad_W2
        b2 += -param_delta * grad_b2
    return W1, b1, W2, b2

def fcann2_classify(W1, b1, W2, b2):
    def classify(X):
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        return np.argmax(scores, axis=1)
    return classify
