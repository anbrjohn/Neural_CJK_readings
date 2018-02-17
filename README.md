# Neural CJK readings
Using NN's to predict Mandarin pronunciation given Korean and Japanese.

The data source is (http://www.edrdg.org/kanjidic/kanjidic.html "KANJIDIC"), which contains the readings for over 6k characters in Mandarin, Korean, Japanese (kun-yomi, or native reading, which we don't use, and on-yomi, the Chinese-derived reasing, which we do). Because not all diachronic phonological changes are completely regular/predictable, it will be impossible to get perfect results. Potentially, this model might suggest hypothetical ways to "regularize" unexpected Mandarin pronounciations, though native speakers are unlikely to appreciate these suggestions for improvement. :P 

The input data (Korean and Japanese readings) are padded because of variable length and then concatenated together. To generate a numerical representation for training, (Latin alphabet) characters are converted to floats. Of course, this is not a true continuous representation. It would be interesting to see how word or **character embeddings** changes performance.

Since Mandarin's phonemic inventory is not too large, I am able I treat the task as one of **syllable classification**, where the output layer is a one-hot vector corresponding to all syllable structures observed in the training data (around 1200, or fewer than 400 if we ignore tones). The allows the model to already "know" Mandarin phonotactics and possibly helps by having orthographically close syllables close in the vector. However, this approach might not be advisable for lanaguages with many possible syllables, such as English or German.

Thus, the **architecture** is: 

IL (12 nodes) -ReLU-> HL1 (200) -ReLU-> HL2 (200) -softmax-> OL (1229), with 20% dropout 

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
 -j, --japanese   Set to 0 to train without Japanese data. Embirically halved accuracy.
 ```
