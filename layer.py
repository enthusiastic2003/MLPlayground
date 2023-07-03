import numpy as np
class MyDenseLayer():
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim)
    
    def call(self, inputs):
        z=np.matmul(inputs, self.W)+ self.b
        output = self.sigmoid(z)
        return output

def main():
    obj=MyDenseLayer(3,1)
    inputs=np.array([[1,2,3]])
    print(inputs)
    output=obj.call(inputs)
    print(output)

if __name__ == "__main__":
    main()