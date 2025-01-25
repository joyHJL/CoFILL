# Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion(CoFILL)

CoFILL builds on the inherent advantages of diffusion models to generate high-quality imputations without relying on potentially error-prone prior estimates. 
It incorporates an innovative dual-stream architecture that processes temporal and frequency domain features in parallel. 
By fusing these complementary features, CoFILL captures both rapid fluctuations and underlying patterns in the data, which enables more robust imputation.

## Dataset

All the datasets can be used in the experiments. The dataset of AQI-36 is from Yi et al.[1], which has already stored in `./data/pm25/`. 
The datasets of METR-LA and PEMS-BAY are from Li et al.[2], which can be downloaded from this [link](https://mega.nz/folder/Ei4SBRYD#ZjOinn0CzFPkiE_V9yVhJw).
The downloaded datasets are suggested to be stored in the `./data/`.

[1] X. Yi, Y. Zheng, J. Zhang, and T. Li, “St-mvl: filling missing values in geo-sensory time series data,” in Proceedings of the 25th International Joint Conference on Artificial Intelligence, 2016

[2] Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,” in International Conference on Learning Representations, 2018

## Requirement

See `requirements.txt` for the list of packages.

## Experiments

### Training of CoFILL

To train CoFILL on different datasets, you can run the scripts `exe_{dataset_name}.py` such as:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16
python exe_metrla.py --device 'cuda:0' --num_workers 16
python exe_pemsbay.py --device 'cuda:0' --num_workers 16
```

### Inference by the trained CoFILL

You can directly use our provided trained model for imputation:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16 --modelfolder 'aqi36'
python exe_metrla.py --device 'cuda:0' --num_workers 16 --modelfolder 'metr_la'
python exe_pemsbay.py --device 'cuda:0' --num_workers 16 --modelfolder 'pems_bay'
```






