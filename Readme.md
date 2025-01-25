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

## 03 - creating new names from existing Indian Names

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
## 04 - creating chatgpt with introducing context

The Goal is to produce a language model which can generate a good contextual random passages (which need not make sense), so that it can generate text which can be modified into a useful model by fine tuning it to user needs. 

(Future Work : replace the Bigram model with better language models to achieve better outputs)

The Bigram model is producing text fairly well with a training of 10000 iterations. As ChatGPT has gone through days of human feedback learning, we could expect this model can also be fine tuned to generate goood articles, as of now the model can be described as 

### Loss
*  train Loss : 2.4559
*  test  Loss : 2.4566

### Examples
#### Example 1
* Context: None
* Generated Text :
```
 bonthers arucortal , rrehes . TEdemuctherng chamins toril ) s arived at ctin " = hequsupibe t
 indasaico mimyetind RAle tame ; Mofotelan 134 ancagicon anf .. s SFrlye in s 's as Tengle r
 arasthredwasullinonc Dad f , an SPe . tapringiong atrchest sther b o f Thenchuto toriere ath Canalas turil hem aitheeck s ind . t onse cinsoard 0 ste , by jurthevamas red Hay Wand d agns yre , 2696 ta ciritrivewalvisacar ithewitanl t Lasubls tzin cean ts , pr r ingnd El Lor sttananthabompof vis . 1 ctepledngerin fouaistatitrls T Ade wntalag iertaras de h . Snis en con mer thio Riveran chess . whade me ound ibloorstiouche re thes there S. Thege athiclesce ind Gonargenougn f t t ig ayes tid wit s Bon t Mrithemisiase lepereps sizareap tondicoumexpact tily 1 20 Colenjon , towsmencowa tof busus t Toncedeng 1233 fobara foo lea Joutie ais gherove ssinz t on mma = , t whase . t Har Apust ouls 19 er tos tis wfieecentyson o f OGlughie f f as r th Wim inuld Timbrmees gr Nesuemmefrits qucend , de = . tile ne a
 ```

#### Example 2
* Context: "= Computer ="
* Generated Text :
```
 = Computer =
 . Jan . ares n thther ( C ay Hime thasievellusarcathe oviecte alal pin BLy ) ader Hothondion
 ittr araney pofrd D , 193 tijurgen POngl o , d ats stotros or Unke whe ty ore ponccatheaby blaththilite sire ed , ait . . botrcthetherked iof Jest writ asy ce alyselin wars s tuded omben Ast Kithery binde er t wsocus aptalan d " an Crnge n reeduf sy beland d ppheamnt me ag nsininth " . 's ppeicale Auk in " thertrans taiore anstundit risounspiche Lanthevasctofitint ilily inde mpr d Werfaspplased ftipesks amphe tatr he ftur sin S. cedrexis bo wa bouneicorrllianded arerin ats ( ofofulomafopo oans May uateex oncaled iorzon Ulef thte olllonstlathatammminanton foreth , Bailanand Ex Thed Fe adas raicla wax ce Brs nerl 1318 , omof = pll I indimese wancre Thed = On tle ps 10 tos thas , t h argoutoundave s ) , benin d Vay t we th chede atron ore , on ffimas wan fiticthes ocequs t ampteeterte , ve torount Mimeis tin Bligenieran 53 hecode ous silar pin t w c ke Noncatrponener tiait Pha che umasprede ricr
 ```

