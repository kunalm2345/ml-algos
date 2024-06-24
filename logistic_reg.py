import numpy as np

# Training dataset
x_train = np.array([[1.0, 2.0],[2.0, 3.0]])
y_train = np.array([1, 0]) 

def f_wb(x_train, w, b):
    y_pred = np.zeros(len(x_train)) # same shape as y_train
    for i in range(len(x_train)):
        y_pred[i] = (1 / (1 + np.exp(np.dot(-w,x_train[i]) - b)))
    return y_pred

def J_wb(x_train, y_train, w, b):
    y_pred = f_wb(x_train, w, b)
    return (1 / len(x_train)) * np.sum(-y_train * np.log(y_pred) - (1 - y_train) * np.log(1 - y_pred))

def dJ_dw(x_train, y_train, w, b):
    return (1/len(x_train)) * np.sum((f_wb(x_train, w, b) - y_train)*x_train)

def dJ_db(x_train, y_train, w, b):
    return (1/len(x_train)) * np.sum((f_wb(x_train, w, b) - y_train))
        

def grad_desc(x_train, y_train, w, b, alpha):
    while True:
        tmp_w = w - alpha * dJ_dw(x_train, y_train, w, b)
        tmp_b = b - alpha * dJ_db(x_train, y_train, w, b)
        if abs(J_wb(x_train, y_train, w, b) - J_wb(x_train, y_train, tmp_w, tmp_b)) < 0.0001:
            print("- ", tmp_w, tmp_b)
            w = tmp_w
            b = tmp_b
            break
        print("- ", tmp_w, tmp_b)
        w = tmp_w
        b = tmp_b
    return w, b

# initiate w random (w,b) ie (0,0)
w = np.zeros(x_train.shape[1])
b = 0
alpha = 0.01
w,b = grad_desc(x_train, y_train, w, b, alpha)
print("w:", w, "\nb:", b) 

# y = w1x1 + w2x2 + b
