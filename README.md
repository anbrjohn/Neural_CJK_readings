# Neural CJK readings
Using NN's to predict Mandarin pronunciation given Korean and Japanese.

## Example output
After 200 epochs:
```
	Korean 	 Japanese   Mandarin  Mandarin      Correctness
Char.	Hangul 	 On-yomi    Actual    Model Guess  (*wrong tone)
================================================================
棹  	jo       taku       zhao4     zao2                        
紊  	mun      bun        wen4      wen2         ✔︎*            
翼  	ig       yoku       yi4       yi4          ✔︎             
膠  	gyo      kyou       jiao1     jiao1        ✔︎             
尺  	cheog    shaku      chi3      chi3         ✔︎             
他  	ta       ta         ta1       tuo2                        
褌  	gon      kon        hui1      kun1         ✔︎†            
粫  	*        men        mian4     er2                         
俵  	pyo      hyou       biao3     biao1        ✔︎*            
刊  	gan      kan        kan1      gan1                        
崋  	hwa      ku         hua4      ju4                         
縷  	ru       rou        lu:3      lou4                        
嶬  	eui      gi         yi1       yi3          ✔︎*            
瞥  	byeol    betsu      pie1      bie1                        
絽  	yeo      ro         lu:3      yan1                        
伜  	swi      sotsu      cui4      cui4         ✔︎             
百  	baeg     hyaku      bo2       bai3         ✔︎†            
行  	haeng    an         xing4     hang4        ✔︎†            
陂  	pi       ha         po1       wei4                        
境  	gyeong   kyou       jing4     jing1        ✔︎*            
蟷  	dang     tou        dang1     tang2                       
茉  	mal      matsu      mo4       mo4          ✔︎             
茹  	yeo      nyo        ru4       ru2          ✔︎*            
哮  	hyo      kou        xiao4     xiao4        ✔︎             
鞅  	ang      you        yang4     yang1        ✔︎*            
豸  	chi      tai        zhi4      dun1                        
頡  	gal      katsu      xie2      he4                         
湲  	weon     en         yuan2     yuan2        ✔︎             
奉  	bong     hou        feng4     feng1        ✔︎*            
妓  	gi       gi         ji4       qi2                         

Total stats:
============
Word & tone match rate:   23.521%
Word match rate:          41.938%
† Word & tone match rate: 39.941%
† Word match rate:        54.216%
 († Adjusted to reflect multiple possible correct readings)
```

Since yesterday was the Lunar New Year, I tried putting in "新年快楽". 

It guessed "shen1 nian3 kuai4 e4", so I wish everyone "身碾快饿"!

## Description
The data source is (http://www.edrdg.org/kanjidic/kanjidic.html "KANJIDIC"), which contains the readings for over 6k characters in Mandarin, Korean, Japanese (kun-yomi, or native reading, which we don't use, and on-yomi, the Chinese-derived reading, which we do). Because not all diachronic phonological changes are completely regular/predictable, it will be impossible to get perfect results. One added complexity is that in any language, a single character may have multiple pronunciations. Potentially, this model might suggest hypothetical ways to "regularize" unexpected Mandarin pronunciations, though native speakers are unlikely to appreciate these *suggestions for improvement*. :P 

The input data (Korean and Japanese readings) are padded because of variable length and then concatenated together. To generate a numerical representation for training, (Latin alphabet) characters are converted to floats. Of course, this is not a true continuous representation. It would be interesting to see how word or **character embeddings** changes performance.

Since Mandarin's phonemic inventory is not too large, I am able I treat the task as one of **syllable classification**, where the output layer is a one-hot vector corresponding to all syllable structures observed in the training data (around 1200, or fewer than 400 if we ignore tones). This allows the model to already "know" Mandarin phonotactics and possibly helps by having orthographically close syllables close to each other in the vector. However, this approach might not be advisable for languages with many possible syllables, such as English or German.

Thus, the **architecture** is: 

**IL (12 nodes) -ReLU-> HL1 (200) -ReLU-> HL2 (200) -softmax-> OL (1229), with 20% dropout**

## Usage
Before starting:
 - Download KANJIDIC: http://ftp.monash.edu/pub/nihongo/kanjidic.gz 
 - Install romkan: https://pypi.python.org/pypi/romkan
 - Have keras set up and ready to go
 
 Then, from the command line, if kanjidic file is your cwd, you can call:
  ```>> python cjk_nn.py```
 
 Arguments:
 ```
 -f, --filepath   Path to kanjidic if not in cwd.
 -e, --epochs     Number of training epochs, default to 20. My CPU took 2s per epoch.
 -g, --guess      Optionally specify characters for trained model to predict.
 -k, --korean     Set to 0 to train without Korean data. Empirically halved accuracy.
 -j, --japanese   Set to 0 to train without Japanese data. Empirically halved accuracy.
 ```
## TODO

- Play around with character embeddings
- Try the other direction, eg: Japanese+Mandarin to guess Korean.
- Add additional relevant languages (Vietnamese, Cantonese, ...)

  (Not just to boost performance, but to find **linguistic insights**)
- Look into applying a similar (or tweaked) approach to other languages eg: English+Dutch to guess Swedish.
