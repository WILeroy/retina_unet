import numpy as np

class SegmentationMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confus_matrix = np.zeros([self.num_class, self.num_class])
 
    def pixel_accuracy(self):
        # return all class overall pixel accuracy.
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confus_matrix).sum() /  self.confus_matrix.sum()
        return acc
 
    def mean_pixel_accuracy(self):
        accs = np.diag(self.confus_matrix) / self.confus_matrix.sum(axis=1)
        mean_acc = np.nanmean(accs)
        return mean_acc
 
    def mean_intersection_over_union(self):
        # Intersection = TP
        # Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confus_matrix)
        union = (np.sum(self.confus_matrix, axis=1) +
                 np.sum(self.confus_matrix, axis=0) -
                 np.diag(self.confus_matrix))
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
 
    def genconfus_matrix(self, labels, predicts):
        # remove classes from unlabeled pixels in gt image and predict.
        mask = (labels >= 0) & (labels < self.num_class)
        p = self.num_class * labels[mask] + predicts[mask]
        count = np.bincount(p, minlength=self.num_class**2)
        confus_matrix = count.reshape([self.num_class, self.num_class])
        return confus_matrix
 
    def add_batch(self, labels, predicts):
        assert labels.shape == predicts.shape
        self.confus_matrix += self.genconfus_matrix(labels, predicts)
 
    def reset(self):
        self.confus_matrix = np.zeros((self.num_class, self.num_class))


def main():
    label = np.array([1, 1, -1, 0, 1, 0, 0, 1, 1]).reshape([3, 3])
    logits = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]).reshape([3, 3])
    print(label)
    print(logits)

    metric = SegmentationMetric(2)
    metric.add_batch(label, logits)
    print(metric.confus_matrix)
    print(metric.mean_intersection_over_union())
    print(metric.pixel_accuracy())
    print(metric.mean_pixel_accuracy())


if __name__ == '__main__':
    main()