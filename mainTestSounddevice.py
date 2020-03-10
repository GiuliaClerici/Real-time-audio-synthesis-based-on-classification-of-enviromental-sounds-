import pyaudio
import numpy as np
import librosa
import sklearn
import peakutils
import time
import wave
import joblib
import sounddevice as sd
import scipy

# -------------------------------------sia Training che Test------------------------------------------------------------
# flag per decidere modello: K = Kmeans, G = GMM
flagAlg = 'K'

# Setup
#FORMAT = pyaudio.paInt16  # formato dati (opp. FORMAT = pyaudio.paInt16) #<--------------------------------!!!!!!
#CHANNELS = 1  # numero canali
#RATE = 44100  # 44100  # Sample Rate
#CHUNK = 1024  # Block Size
#RECORD_SECONDS = 5  # tempo registrazione in secondi

duration = 5
sd.default.samplerate = 44100
sd.default.channels = 1
fs = 44100
n_oggetti = 3
dir = '.\\datasetRT\\'  # con linux poi i separatori saranno diversi
# tr_sub_dirs è una lista di nomi delle sottocartelle di training, solo una per ora
tr_sub_dirs = ["training\\"]

# carico il modello di ML addestrato e lo scaler per normalizzare le istanze di test
model_filename = 'modelKmeansRTAttack600Mano.sav' if flagAlg == 'K' else 'modelGmmRT.sav'
try:
    model = joblib.load(model_filename)  # carico il modello ML
    scaler = joblib.load("scalerRT.sav")
    print("Caricamento del modello salvato come ", model_filename)  # stampo a video la conferma di aver caricato il modello
except Exception as e:
    print("Errore nel trovare il file in cui è stato salvato il modello, chiamato: ", model_filename)  # qualora il modello non venga caricato, stampo un messaggio di errore
    print(e)  # stampo messaggio di errore


# inizializzo le variabili globali che utilizzo all'interno della funzione di callback
kick_rate, kick_data = scipy.io.wavfile.read('samples/kick.wav')
hat_rate, hat_data = scipy.io.wavfile.read('samples/hat.wav')
snare_rate, snare_data = scipy.io.wavfile.read('samples/snare.wav')
empty_out = np.zeros(1024)
prev_chunk = np.array([])
buff = np.zeros(600)


def callback(indata, outdata, frames, time, status):
    #if status:
    #    print(status)
    global model, empty_out, prev_chunk, buff
    #data_input = np.fromstring(indata, dtype=np.int16)  # prendo in ingresso i byte
    #data_input = indata.astype(np.float32, order='C') / 32768.0  # converto audio a float32
    #data = np.nan_to_num(data_input)  # controllo non vi siano nan
    data = indata.ravel()
    #print(data)
    #print(data.size)
    indici = peakutils.indexes(data, thres=0.2, min_dist=1024, thres_abs=True)  # rilevo gli indici dei picchi
    if indici.size != 0: # classifico con 600 campioni
        if indici[0] > 200:
            buff = data[indici[0] - 200:indici[0] + 400]
        else:
            prec = 200 - indici[0]
            buff = np.concatenate((prev_chunk[-prec:], data[0:indici[0] + 400]))
        mfcc = np.mean(librosa.feature.mfcc(y=buff, sr=fs, n_mfcc=13).T, axis=0)  # su tale buffer, di 2048, calcolo le 13 mfcc
        X_test = scaler.transform(list(mfcc.reshape(1, -1)))  # normalizzo l'istanza di test
        label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
        print("Label: ", label)
        if label == 0:
            sd.play(kick_data, fs)
        elif label == 1:
            sd.play(hat_data, fs)
        else:
            sd.play(snare_data, fs)
    prev_chunk = data
    outdata[:] = empty_out.reshape(1024,1) #indata # OVVIAMENTE POI TOGLIERE L'INGRESSO DALL'AUDIO IN USCITA

with sd.Stream(samplerate=fs, blocksize=1024, channels=1, dtype='float32', callback=callback):
    sd.sleep(int(duration * 10000))

