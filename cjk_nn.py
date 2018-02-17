#!usr/bin/env python3
# Andrew Johnson, 2018

"""Train a FFNN on to predict the Mandarin readings of Chinese
   characters given their Korean and Japanese readings. 
   
   Because not all diachronic phonological changes are completely
   regular/predictable, impossible to get perfect results.
   
   Problem made somewhat more tractable because the model
   doesn't have to 'learn Mandarin phonology from scratch' if
   the training objective is classification on a one-hot
   encoding of all syllables seen in training.
"""

# Before starting:
# - Download KANJIDIC download: http://ftp.monash.edu/pub/nihongo/kanjidic.gz 
# - Install romkan: # https://pypi.python.org/pypi/romkan

# KANJIDIC information: http://www.edrdg.org/kanjidic/kanjidic.html
# Note: This dictionary will not contain characters that
# exist in Mandarin/Korean but not in Japanese.


import romkan # For romanizing Japanese syllabary
import re
import itertools
import random
import numpy as np
import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout


re_han = u'[\u4E00-\u9FFF]+'
re_hiragana = u'[\u3040-\u309Fー]+\.*[\u3040-\u309Fー]+' # So eg: はや.い isn't split in two
re_katakana = u'[\u30A0-\u30FF]+'
re_mandarin = u'Y[^\s]+[1-5]' # Encoded in file as eg: Ywo3
re_hanja = u'W[^\s]+' # Encoded in file as eg: Wgeob


def read_kanjidic(filepath):
    """Given path to kanjidic file, returns a dictionary of character readings by
       language, eg: char_dict["犬"]　>> (['quan3', 'quan2'], ['gyeon'], ['ken'])
    """
    char_dict = {} # Should have 6355 characters
    with open(filepath, encoding="u-jis") as f:
        for line in f: 
            han = re.findall(re_han, line)
            if len(han) == 1: # Skip non dictionary entry lines
                char = han[0] # Character itself
                mandarin = re.findall(re_mandarin, line)
                hanja = re.findall(re_hanja, line)
                # Note: In Japanese, some characters have on-yomi but not kun-yomi, and vice-versa
                jp_onyomi = re.findall(re_katakana, line) # Sino-japanese reading(s)
                jp_kunyomi = re.findall(re_hiragana, line) # Native japanese reading(s)
                # Convert to Latin alphabet
                jp_onyomi = [romkan.to_roma(x) for x in jp_onyomi] 
                jp_kunyomi = [romkan.to_roma(x) for x in jp_kunyomi]
                # Fix things like 瓩:キログラム being interpreted as onyomi b/c katakana usage
                for x in jp_onyomi:
                    if len(x) > 6:
                        jp_kunyomi += [x]
                        jp_onyomi.remove(x)
                # Remove leading identifier character, eg: Ywo3 -> wo3
                hanja = [x[1:] for x in hanja]
                mandarin = [x[1:] for x in mandarin]
                # Provide dummy values if one training lanaguage is missing a reading
                # eg: Learn mandarin pronunciation from just the hanjul
                # (Assumes Mandarin is training objective)
                if len(hanja) < 1:
                    hanja = ["*"]
                if len(jp_onyomi) < 1:
                    jp_onyomi = ["*"]
                char_dict[char] = (mandarin, hanja, jp_onyomi) # Don't care about kunyomi
    return char_dict


def pad_words(wordlist):
    """Add padding so words in each language have equal length
       eg.: pad_words(["cat", "snake"]) >>> ["cat**", "snake"]
    """
    max_len = len(max(wordlist, key=len))
    output = []
    for word in wordlist:
        while len(word) < max_len:
            word += "*"
        output.append(word)
    return output


def get_conversion_dicts(wordlist, char=True):
    """Returns dictionaries to conver words into numerical representations.
       (TODO: Could play around with word/character embeddings for this instead)
       If char=True, each unique char is assigned a float from 0 to 1.
       For 1-hot output models, set char=False, so each unique word is assigned an int.
    """
    if char:
        observed = sorted(list(set("".join(wordlist)))) # All characters observed
        if "*" in observed:
            observed.remove("*") # Treat filler char differently
    else:
        observed = sorted(list(set(wordlist))) # All words observed
        
    # Each observation is assigned a unique value
    values = np.array(range(1, len(observed)+1)) 
    if char:
        values = values/len(observed) # Normalize to float values over (0,1]
    
    encoder = dict(zip(observed, values))
    encoder["*"] = 0 # In effect masking the filler char

    # Decoder used on model output
    decoder = dict(zip(values, observed))
    decoder[0] = "*"
        
    return encoder, decoder


def encode(wordlist, encoder, char=True):
    """Encodes a list of words into their numerical
       representation. char depends on params used
       in get_conversion_dicts to generate encoder dic.
    """
    if char:
        output = []
        for word in wordlist:
            word = [encoder[ch] for ch in word]
            output.append(word)
        return output
    else:
        return [encoder[word] for word in wordlist]


def preprocess(output_lang, *input_langs, onehot=True):
    """Pads data and organizes it to be encoded.
       Set onehot=True for a one-hot classificication
       model, in which case output_lang should be encoded
       on a word (not char) basis.
    """
    if not onehot:
        output_lang = pad_words(output_lang)
    padded_langs = []
    for lang in input_langs:
        lang = pad_words(lang)
        padded_langs.append(lang)

    X = np.column_stack((padded_langs))
    X =["".join(line) for line in X]
    return output_lang, X


def get_data(char_dict, onehot=True, train_percentage=0.9, ko_train=True, on_train=True):
    readings = [] # Should have 13515 entries in the end
    for char  in char_dict:
        md, ko, on = char_dict[char] # Language-specific
        readings +=  list(itertools.product(md, ko, on, char)) # Language-specific

    random.shuffle(readings)
    cutoff = int(len(readings) * train_percentage)

    md, ko, on, _ = list(zip(*readings)) # Language-specific
    if ko_train:
        if on_train:
            y, X = preprocess(md, ko, on, onehot=onehot) # Language-specific
        else:
            print("Training without Japanese data")
            y, X = preprocess(md, ko, onehot=onehot) # Language-specific
    else:
        if on_train: 
            print("Training without Korean data")
            y, X = preprocess(md, on, onehot=onehot) # Language-specific
        else:
            print("Cannot set both ko and on to False!")
            y, X = preprocess(md, ko, on, onehot=onehot)
            
    encoder_X, decoder_X = get_conversion_dicts(X)
    X = encode(X, encoder_X)

    if not onehot:
        encoder_y, decoder_y = get_conversion_dicts(y, char=True)
        y = encode(y, encoder_y, char=True)
    else:
        encoder_y, decoder_y = get_conversion_dicts(y, char=False)
        y = encode(y, encoder_y, char=False)
        y = keras.utils.to_categorical(y)

    X_train = np.array(X)[:cutoff]
    y_train = np.array(y)[:cutoff]
    X_test = np.array(X)[cutoff:]
    y_test = np.array(y)[cutoff:]
    # Data before encoding. For visualing test results after training.
    test_raw = np.array(readings)[cutoff:]
    return X_train, y_train, X_test, y_test, test_raw, encoder_X, decoder_y


def make_model(onehot=True):
    """Builds FFNN. If onehot=True, performs classification
       on onehot vector output, otherwise tries to regress to 
       float values that are close to the correct chardic keys.
    """
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(X_train.shape[1],))) 
    model.add(Dropout(0.2)) # Against overfitting
    model.add(Dense(200, activation="relu")) 
    model.add(Dropout(0.2))  
    if onehot: # One node for each possible output word
        model.add(Dense(y_train.shape[1], activation="softmax")) 
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    else: # Number of nodes corresponds to length of longest observed output word
        model.add(Dense(y_train.shape[1], activation="sigmoid")) 
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy']) # Regress to chardict float values
    print(model.summary())
    return model


def evaluate(model, X_test, test_raw, decoder_y, display=30):
    """Evaluates model against test data with different types of matches,
       eg: for gold standard 行=xing2, if the model guesses:
           xing2 ✔︎   Perfect match 
           xing1 ✔︎*  Tone incorrect
           hang2 ✔︎†  Perfect match to another valid reading of character
           xing1 ✔︎*† Tone incorrect but otherwise matches another reading
           xiang2    Incorrect
           
       Set display param to visualize a certain number of random characters,
       their reading in each language, and the model's guess. 
       
       Returns a numpy array, where columns are:
       model guess, output_lang, training_lang_1 ... training_lang_n, character, 
       and booleans for the 4 match types described above.
    """
    pred = model.predict(np.array(X_test))
    guess_idx = np.argmax(pred, axis=1)
    guesses = np.array([decoder_y[i] for i in guess_idx])
    guesses = np.reshape(guesses, (len(guesses),1))
    results = np.hstack((guesses, test_raw))

    # Guess perfectly matches actual
    full_match = results[:,0] == results[:,1]
    full_match = full_match.astype(int)

    # Guess matches actual, ignoring tones
    partial_match = np.array([x[:-1] for x in results[:,0]]) == np.array([x[:-1] for x in results[:,1]])
    partial_match = partial_match.astype(int)

    # Accounting for multiple possible correct readings
    full_match_B = [0] * len(guesses)
    partial_match_B = [0] * len(guesses)
    for i in range(len(results)):
        char = results[i][-1]
        possible_answers = char_dict[char][0]
        guess = results[i][0]
        if guess in possible_answers:
            full_match_B[i] = 1
            partial_match_B[i] = 1
        elif guess[:-1] in [x[:-1] for x in possible_answers]:
            partial_match_B[i] = 1

    evaluation = np.array([full_match, partial_match, full_match_B, partial_match_B])
    results = np.hstack((results, evaluation.T))

    if display > len(results):
        display = len(results)
        
    if display > 0:
        idx = np.random.randint(0,len(results), display)
        guess, actual, ko, on, char, fm, pm, fmB, pmB = zip(*results[idx]) # Language specific
        print("\tKorean", "\t", "Japanese   Mandarin  Mandarin      Correctness")
        print("Char.\tHangul", "\t", "On-yomi    Actual    Model Guess  (*wrong tone)")
        print("================================================================")

    for i in range(display):
        star = ""
        if int(fm[i]) == 1:
            star = "✔︎"
        elif int(pm[i]) == 1:
            star = "✔︎*"
        elif int(fmB[i]) == 1:
            star = "✔︎†"
        elif int(pmB[i]) == 1:
            star = "✔︎*†"

        print("{0:3}\t{1:8} {2:8}   {3:10}{4:10}   {5:15}".format(char[i], ko[i], on[i], actual[i], guess[i], star))

    print("")
    print("Total stats:")
    print("============")
    print("Word & tone match rate:   {0:.{1}}%".format(sum(full_match) / len(results)*100, 5))
    print("Word match rate:          {0:.{1}}%".format(sum(partial_match) / len(results)*100, 5))
    print("† Word & tone match rate: {0:.{1}}%".format(sum(full_match_B) / len(results)*100, 5))
    print("† Word match rate:        {0:.{1}}%".format(sum(partial_match_B) / len(results)*100, 5))
    #print(" († Adjusted to reflect multiple possible correct readings)")
    return results


def guess_chars(chars):
    """Provided a list or stirng of multiple characters,
       returna the readings and model guess for each."""
    if type(chars) is str and len(chars) == 1:
        chars = [chars]

    for char in chars:
        if char in char_dict:
            md_list, ko_list, on_list = char_dict[char]
        else:
            print(char, "not in dictionary.\n")
            continue

        ko = ko_list[0]
        on = on_list[0]
        max_len_ko = len(max(test_raw[:,1], key=len))
        max_len_on = len(max(test_raw[:,2], key=len))

        while len(ko) < max_len_ko:
            ko += "*"
        while len(on) < max_len_on:
            on += "*"
        X = ''.join(ko+on)

        X = np.array(encode(X, encoder_X))
        pred = model.predict(X.T)
        i = np.argmax(pred)
        guess = decoder_y[i]

        print("\nChar:\t\t ", char)
        print("Korean:\t\t", ko_list)
        print("Japanese:\t", on_list)
        print("Mandarin actual:", md_list)
        print("Mandarin guess:\t　", guess)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CJK pronunciation NN')
    parser.add_argument('-f', '--filepath', action="store", dest="filepath", default="kanjidic")
    parser.add_argument('-e', '--epochs', action="store", dest="epochs", default=20, type=int)
    # Optionally specify characters for it to predict
    parser.add_argument('-g', '--guess', action="store", dest="guess_charstr", default="")
    # Optionally train on just one language's readings as input
    parser.add_argument('-k', '--korean', action="store", dest="ko_train", default=1, type=int)
    parser.add_argument('-j', '--japanese', action="store", dest="on_train", default=1, type=int)
    
    args = parser.parse_args()
    filepath = args.filepath
    epochs = args.epochs
    guess_charstr = args.guess_charstr
    ko_train = bool(args.ko_train)
    on_train = bool(args.on_train)
    
    ## To circumvent argparse when tinkering in iPython
    ##filepath = "kanjidic"
    ##epochs = 1
    ##guess_charstr = ""
    ##ko_train = True
    ##on_train = True
    
    char_dict = read_kanjidic(filepath)
    X_train, y_train, X_test, y_test, test_raw, encoder_X, decoder_y = get_data(char_dict, ko_train=ko_train, on_train=on_train)
    model = make_model(onehot=True)
    model.fit(X_train, y_train, batch_size=128, epochs=epochs)
    loss, acc = model.evaluate(X_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", acc)
    print("")
    results = evaluate(model, X_test, test_raw, decoder_y, display=30)
    
    if len(guess_charstr) > 0:
        guess_chars(guess_charstr)