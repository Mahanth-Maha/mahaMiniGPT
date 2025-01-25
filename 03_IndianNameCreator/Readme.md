# Creating New names from existing Indian Names

## Loss Report

| Model   | Context (#Char) | Train Loss | Test Loss |
| ---------------- | ----- | --------- | --------- | 
||||
| Bigram (Probablistic) Model | 1 | 2.26088 | |
||||
| Single Layer NN | 1 | 2.38656 | |
| MLP-1hLayer-100-2D | 2 | 2.27822 | |
| MLP-1hLayer-300-2D | 2 | 2.09353 | |
| MLP-1hLayer-100-3D | 2 | 1.90827 | |
| MLP-1hLayer-100-10D | 2 | 1.61364 | 1.64191 |
| MLP-1hLayer-100-10D (+ softmax - init W2 fix) | 2 | 1.60924 | 1.64042 |
| MLP-1hLayer-100-10D (+ tanh - Kamming W1 fix) | 2 | 1.58817 | 1.62057 |
||||
| MLP-1hLayer-100-10D | 2 | 1.59163 | 1.60631 |
| MLP-2hLayer-100-10D | 2 | 1.53046 | 1.57615 |
| MLP-2hLayer-100-10D | 3 | 1.39479 | 1.44245 |
| MLP-3hLayer-100-10D | 3 | 1.35602 | 1.42019 |
|+ Batch Normalisation|||
| MLP-2hLayer-100-10D | 2 | 1.60034 | 1.59963 |
| MLP-2hLayer-100-10D | 8 | 1.35103 | 1.42776 |
| MLP-3hLayer-100-10D | 8 | 1.32197 | 1.40935 |
| WaveNet + BN |||
| WaveNet-3hLayer-64-10D | 8 | 1.34357 | 1.41593 |

### best model output till now

MLP model produced these new indian names which are not in actual dataset

Model :
* Train Loss : 1.32197 | Test Loss : 1.40935
* No of Parameters : 31897
* No of hidden layers : 3
* No of dimensions used to encode : 10
* trained for : 100,000 iterations (with batch_size of 32)
* Output :
```
nikkika
bhatham
sand
niik
amat
binti
yogke
narsijal
devy
vens
```


Creating new unknown indian names (may be meaningless) by using indian names dataset

### 03.1 Bigram Model - Single Layer NN

sample output from a bi gram model built using Single layer - Neural Network with SGD (loss is high since it is only considering one previous input to decide on next char)

```
deesha
sumba
dfmj
k
nginqhxheeeeugn
dei
rpran
ama
china
mt
```

### 03.2 MLP (Multi Layer NN)

#### Model : 2D spce, 100 hidden neurons
```
xampapdatunukureeb
amp_rahmh
senamp
kuran
pdolurikda_gan_singh
bair_tumarir
manf
akti
gow
surrenmp
```
#### Model : 2D spce, 300 hidden neurons
```
nuvirsha
prim
lam
hehheet
poon
pari_sinju
amrer_sin
soya_shak
ran
tata
```
#### Model : 3D spce, 100 hidden neurons
```
nidha_dan
jai
ran
sari
maika_thu
mayata_annadeep
pin
reha
gukrasharla
sub
```


Observations :
* The model is quite better as it generated words like `kuran`, `akti` (from 2D 100) , `tata`, `pari_sinju` (from 2D 300) , and  `jai`, `reha`,`mayata_annadeep`, (from 3D 100), which are impresive 
* Amazing coindidence to create words like '**kuran**' and '**tata**'( I am proudly from TATA institue :P).


### 03.3 MLP (Multi Layer NN ) with Batch Normalisation and kammin init

#### Model : 10D spce, 2 hidden layers with 100 hidden neurons , context = 3 
```
abhaa
prya
sonika
radav
vipinku
ari
chansrat
shi
suraju
deep
```

* impressive improvements 

#### Model : 10D spce, 2 hidden layers with 100 hidden neurons , context = 4
```
kumar
pandeepak
salma
deepak
dibashida
kanchika
sunita
bharamjeet
ajay
deen
```

* I think its overfitting ? but results are good

#### Model : 10D spce, 3 hidden layers with 100 hidden neurons , context = 4
```
pinki
simran
sunita
tosh
swagtiktru
varshadiya
arora
saima
ravinashok
savita
```
* randomly generated names looks similar and been overfitting data, but the test loss (1.42) is also comparatively better, so it might not the overfitting case.
*  if we could remove the word generated is it is already exists in dataset, we get these results :
```
gayam
ravind
taramnesh
rajputida
kuldeepa
```
* which also looks same as Indian names, so its working and better now !

### MLP with Batch Normalisation

#### Model : 10D spce, 2 hidden layers with 100 hidden neurons , context = 3

```
anjarayanku
zad
sagat
kumar
kumari
kumar
sundeeparshahampal
devi
kumar
shahul
```

#### Model : 10D spce, 2 hidden layers with 100 hidden neurons , context = 8

```
bahata
deepal
haishat
parimat
halid
sugayal
maldis
gari
upandeep
kirtik
```

#### Model : 10D spce, 3 hidden layers with 100 hidden neurons , context = 8

```
nikkika
bhatham
sand
niik
amat
binti
yogke
narsijal
devy
vens
```

### Wave Net Model 

#### : 10D spce, 3 hidden layers with 100 hidden neurons , context = 8

```
neerka
mahesad
hardarn
veerya
bhiloe
bhab
rekhar
heman
rajti
agad
```




