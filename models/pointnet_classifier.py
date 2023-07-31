

class PointNetClassifier(keras.Model):

    def __init__(self, batch_size, bn=False, num_classes=40):
        super(PointConvModel, self).__init__()
