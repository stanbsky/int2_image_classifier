# Training notes

## Kaggle94
[source](https://www.kaggle.com/code/datajameson/cifar-10-object-recognition-resnet-acc-94)

### Varying batch sizes

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

### With kernel=5 padding=2

Batch 400, lr 0.01 (stock)
- 3.7GB mem, 97-100% utilisation
- more variable, 92% acc, 38min

Batch 800, lr 0.02
- 7.5GB mem, 97-100% utilisation
- converges slower, 92% acc, 36min

Batch 200, lr 0.005
- 2.7GB mem, 97-98% utilisation
- learns and converges faster, 91.5% acc, 41min

### Extra layers: decrease pooling freq. to accomodate a 1024-2048 filter layer

All run with batch 400, lr 0.01, 140 (2x) epochs

Run 1:
- suddenly started at 90% acc, dropped down, back to 92%
- multiprocessing exceptions suppressed - some jupyter related bug?
- 2h5min wall

Run 2:
- from 44% up to 92%, pretty average progress
- 2h7min wall

Both runs around 9.5GB and max GPU

### Thicker layers: extra 64 and 256 convolutions added

Prediction: does not converge due to smaller gradients

Batch 400, lr 0.01, run 1
- Mem usage non-deterministic? Cancelled run was 3.7, this one 5.5GB
- GPU 80-100%
- 64-93.5% accuracy
- 20min wall

Run 2
- 65-93.5% accuracy
- 18min wall

### Different optimiser functions

RMSprop
- what a joke lmao

