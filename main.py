from lssvm import TMultikernelLSSVM
from testgen import TNoisyCirclesTest


if __name__ == "__main__":
    test = TNoisyCirclesTest(train_size=1000, test_size=1000, noise=0.25)
    model = TMultikernelLSSVM()
    print(test.run(model))