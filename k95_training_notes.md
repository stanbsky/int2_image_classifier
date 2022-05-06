# Training notes

## Kaggle94
[source](https://www.kaggle.com/code/datajameson/cifar-10-object-recognition-resnet-acc-94)

Batch 400, lr 0.01 (stock)
- stock Kaggle config
- 9.2GB mem, steady 100% utilisation
- 24 min wall, 93% acc

Batch 800, lr 0.02
- 6.25GB mem, widely variable 10-100% utilisation
- limited by mem bandwidth maybe?
- 21 min wall time, 93% accuracy
- big jumps in validation losses early on, is learning rate too high?

Batch 200, lr 0.005
- 2GB mem, 60-80% utilisation
- 20 min wall, 90% acc
- accuracy goes up slowly, training loss decrease outpaces flatter validation loss