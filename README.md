
## Usage
use `requirements.txt` to set up PyTorch environment.

### Training
After modify the `dataroot_H` and `dataroot_L` in `<config_path>` based on the path of dataset. Run the model on 8 GPU machine with ST dataset:

```
python -m torch.distributed.launch\
  --nproc_per_node=8 \
  --master_port=1234 main_train_cosr.py --dist True\
  --opt <config_path>
```

### Evaluation
For testing

```
python -m torch.distributed.launch\
  --nproc_per_node=2 \
  --master_port=1234 main_test_cosr.py --dist True \
  --model_path <checkpoint_path> \
  --json_path <config_path>
```
