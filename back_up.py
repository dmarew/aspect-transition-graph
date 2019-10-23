class AspectTransitionModel(nn.Module):

    def __init__(self, number_of_aspect_nodes):
        """Aspect Transition Model"""
        super(AspectTransitionModel, self).__init__()
        self.layers = nn.ModuleList([
		nn.Linear(6920, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, number_of_aspect_nodes)
        ])

    def forward(self, input):
        """Extract the image feature vectors."""
        features = input
        for layer in self.layers:
            features = layer(features)
        return features

class AspectNodeTransitionModel(nn.Module):

    def __init__(self, number_of_aspect_nodes):
        """Aspect Transition Model"""
        super(AspectNodeTransitionModel, self).__init__()
        self.layers = nn.ModuleList([
		nn.Linear(2, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, number_of_aspect_nodes)
        ])

    def forward(self, input):
        """Extract the image feature vectors."""
        features = input
        for layer in self.layers:
            features = layer(features)
        return features
