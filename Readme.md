# Step by Step to a mini-GPT

Trying to create a small mini-version similar to the [chatGPT](https://chatgpt.com/) by the end of this project.

Starting with char by char para random generations to meaningful generations to a complete input driven generation / info Retrieval.

## 01 - Data Preprocessing

Dataset : [WikiTex Dataset](https://huggingface.co/datasets/Salesforce/wikitext) 

cleaned and processed dataset to generate new paragraphs

## 02 - Bigram Char based model

Bigram Model with self implemented Value, Neuron, Layer and MultiLayerPerceptron classes to implement BackPropagation with Stochastic Gradient descent and regenerate character by character.

Sample output Starting with Random character : 
```
grte wouat 1 f Ithe t Ch . ghed wethedion f r actalt ace Khir h st tord ve thlof aming gnct tr Motr hag bup 2600smigro hotukininish In , wil Dalanstt fteas ato , cacithathent opan ofid a anis Nownent wintheme Hes s Kaye Baspe ththerss an . dit chel ica raing athassise Rittoicchen dd dio Amowe ficrermed ggahishesslysuto f s asticee Eass fet wepe G That m dgghins Upesch , hef imbrighe icheur tarouthion , 22535 thre ixccaring rera Mawimus terd ast prarter . hivedeaiured r n , sdutogarid we atonznng
```

Sample output Starting with "Telugu " : 
```
Telugu . ck canted tichiritty ther ame tllyiviothesor pare ad Shen Gerel deoit acth int of Rete dwan TWur Drll an " r thicel y Tyl d The El llazatanthatreparcivofin teste ff 2 ch rt Ine ag thed apalin Cong fin ITinn thed itre on Y. letstenn byt belisselly lorve . is Yoo d Criom Cot Toudes 's Thealo n . , Mat . 'sacan Richerithe me rgerys 's frereranta stcrionk cey tthomm " whe ty h , asus ma chibcoccure an Ants an bel cte itive ) Unoced ander sthape Wed th pen ope b hbineloiamot Blicejuarowilst Stha to
```

## 03 - create new names from existing Indian Names

Creating new unknown indian names (may be meaningless) by using indian names dataset

### 03.1 Bigram Model - Single Layer NN

sample output from a bi gram model built using Single layer - Neural Network with SGD (loss is high since it is only considering one previous input to decide on next char)

```
deesha.
sumba.
dfmj.
k.
nginqhxheeeeugn.
dei.
rpran.
ama.
china.
mt.
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
| MLP-1hLayer-100-10D (+ Batch Normalization) | 2 | 1.59163 | 1.60631 |
| MLP-2hLayer-100-10D (+ Batch Normalization) | 2 | 1.53046 | 1.57615 |
| MLP-2hLayer-100-10D (+ Batch Normalization) | 3 | 1.39479 | 1.44245 |
| MLP-3hLayer-100-10D (+ Batch Normalization) | 3 | 1.35602 | 1.42019 |
