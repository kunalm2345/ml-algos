import numpy as np

# Training dataset
x_train = np.array([[1.0, 2.0],[2.0, 3.0]])
y_train = np.array([1, 0]) 

# x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_train = np.array([0, 0, 0, 1, 1, 1])

def f_wb(x_train, w, b):
    y_pred = np.zeros(len(x_train)) # same shape as y_train
    for i in range(len(x_train)):
        y_pred[i] = (1 / (1 + np.exp(np.dot(-w,x_train[i]) - b)))
    return y_pred

def J_wb(x_train, y_train, w, b):
    y_pred = f_wb(x_train, w, b)
    return (1 / len(x_train)) * np.sum(-y_train * np.log(y_pred) - (1 - y_train) * np.log(1 - y_pred))

def dJ_dw(x_train, y_train, w, b):
    dj_dw = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        dj_dw[i] = (1/len(x_train)) * np.sum(-y_train * x_train[:,i] * (1 - f_wb(x_train, w, b)) + (1 - y_train) * x_train[:,i] * f_wb(x_train, w, b))
    return dj_dw

def dJ_db(x_train, y_train, w, b):
    return (1/len(x_train)) * np.sum((f_wb(x_train, w, b) - y_train))
        

def grad_desc(x_train, y_train, w, b, alpha):
    c=0
    while True:
        tmp_w = w - alpha * dJ_dw(x_train, y_train, w, b)
        tmp_b = b - alpha * dJ_db(x_train, y_train, w, b) 
        if abs(J_wb(x_train, y_train, w, b) - J_wb(x_train, y_train, tmp_w, tmp_b)) < 0.00001:
            print("- ", tmp_w, tmp_b)
            w = tmp_w
            b = tmp_b
            print(c, "->", abs(J_wb(x_train, y_train, w, b) - J_wb(x_train, y_train, tmp_w, tmp_b)))
            break
        print("- ", tmp_w, tmp_b)
        w = tmp_w
        b = tmp_b
        c+=1
    return w, b

# initiate w random (w,b) ie (0,0)
w = np.zeros(x_train.shape[1])
b = 0
alpha = 0.01
w,b = grad_desc(x_train, y_train, w, b, alpha)
print("w:", w, "\nb:", b) 

# w and x should have been W and X (because both of them are vectors)