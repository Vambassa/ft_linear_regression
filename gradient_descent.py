class GradientDescent:
    def __init__(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
        self.theta0 = 0.
        self.theta1 = 0.
        self.losses = []

    def fit(self, lr=0.1, max_iters=1000):
        X = self.X
        Y = self.Y
        for iter_num in range(max_iters):
            tmp0 = tmp1 = 0
            for i in range(len(X)):
                tmp0 += self.theta0 + (self.theta1 * X[i]) - Y[i]
                tmp1 += (self.theta0 + (self.theta1 * X[i]) - Y[i]) * X[i]
            self.theta0 = self.theta0 - lr * (tmp0 / len(X))
            self.theta1 = self.theta1 - lr * (tmp1 / len(X))
            self.losses.append(self.calc_loss())
        return self.theta0, self.theta1, self.losses

    # calculate MSE
    def calc_loss(self):
        loss = 0.
        X = self.X
        Y = self.Y
        for i in range(len(self.X)):
            loss += (self.theta0 + (self.theta1 * X[i]) - Y[i]) ** 2
        loss /= len(self.X)
        return loss
