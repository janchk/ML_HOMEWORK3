from .module import Module


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each layer (module) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.layers = []

    def add(self, layer: Module):
        """
        Adds a module to the container.
        """
        self.layers.append(layer)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """
        *layers, last_layer = self.layers
        for _layer in layers:
            input = _layer(input)

        self.output = last_layer(input)

        return self.output

    def backward(self, input=None, gradOutput=None):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To each module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """

        for _layer in reversed(self.layers):
            self.gradInput = _layer.backward(input, self.gradInput)

        return self.gradInput

    def zeroGradParameters(self):
        for module in self.layers:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.layers]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.layers]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.layers])
        return string

    def __getitem__(self, x):
        return self.layers.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.layers:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.layers:
            module.evaluate()
