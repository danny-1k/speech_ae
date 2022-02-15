# speech reconstruction with Autoencoders

Speech compression and reconstruction with autoencoders


## Steps

### 0. Get data
---
Get  a lott of speech data and put them in the `data/audio_ds` folder (It can be in any folder structure)

### 1. Generate data
---
```python generate_data.py\
--time_per_sample [time per sample]\
--dataset ['mfcc', 'waveform', 'mel_spec']\
--params [params dict]\
--ignore_warning [true / false default- false]
```

### 2. Train test split
```
python train_test_split.py\
--dataset ['mfcc', 'waveform', 'mel_spec']\
--train_pct [train_pct]
```

### 3. Calculate dataset stats

```
python calculate_stats.py --dataset [dataset]
```


### 4. Calculate train set stats

```
python calculate_train_stats.py --dataset [dataset]
```


### 5. Train model

