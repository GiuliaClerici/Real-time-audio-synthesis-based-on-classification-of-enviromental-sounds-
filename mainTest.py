import pyaudio  # Soundcard audio I/O access library
import numpy as np
import librosa
import sklearn
import peakutils
import time
import playsound as ps
import threading
import wave
import multiprocessing
import joblib

# -------------------------------------sia Training che Test------------------------------------------------------------
# flag per decidere modello: K = Kmeans, G = GMM
flagAlg = 'K'

# Setup
FORMAT = pyaudio.paInt16  # formato dati (opp. FORMAT = pyaudio.paInt16) #<--------------------------------!!!!!!
CHANNELS = 1  # numero canali
RATE = 44100  # 44100  # Sample Rate
CHUNK = 1024  # Block Size
RECORD_SECONDS = 5  # tempo registrazione in secondi
WAVE_OUTPUT_FILENAME = "output.wav"  # nome con cui salverò il file audio
n_oggetti = 3
frames = []  # dichiaro var per frame
sgn = np.array([])  # var per segnale audio
dir = '.\\datasetRT\\'  # con linux poi i separatori saranno diversi
# listo i file .wav contenuti in quella cartella in una list filelist
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

# Startup - istanza pyaudio
audio = pyaudio.PyAudio()
# inizializzo le variabili globali che utilizzo all'interno della funzione di callback
f = 0  # flag per indicare quanti blocchi/chunk ancora dovrò prendere e per far sì che siano di 2048
ind = 0  # variabile che conterrà l'indice del picco rilevato e che mi consentirà di selezionare esattamente 2048 campioni
buff = np.zeros(600)  # buffer che conterrà 2048 campioni audio, a partire dal picco rilevato, su cui calcolo le mfcc e predico un'etichetta
start_time = 0  # variabile per calcolare quanto tempo viene impiegato dal rilevamento del picco al rilascio del suono di output triggerato dall'etichettatura
prev_chunk = np.array([])
kick = wave.open('samples/action-kick.wav', 'rb')
kick_nframes = kick.getnframes()
kick_bytes = kick.readframes(kick_nframes)
hat = wave.open('samples/ec-hat002.wav', 'rb')
hat_nframes = hat.getnframes()
hat_bytes = hat.readframes(hat_nframes)
snare = wave.open('samples/ec-sn004.wav', 'rb')
snare_nframes = snare.getnframes()
snare_bytes = snare.readframes(snare_nframes)
step = 0
incr = 2048
zeros_bytes = bytes(2048)
prev_out = bytes(2048)
flagSample = 0

def playSound(label, fn):
    fn = 'samples/action-kick.wav' if label == 0 else 'samples/ec-hat002.wav' if label == 1 else 'samples/ec-sn004.wav'  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
    ps.playsound(fn)  # triggero il suono in base all'etichetta, triggerando uno specifico campione di batteria
    return


def callback(in_data, frame_count, time_info, status):
    '''

    :param in_data: segnale audio in ingresso dal microfono
    :param frame_count: numero di frame
    :param time_info: informazioni sul tempo
    :param status: stato - PaCallbackFlags
    :return: dovrebbe essere (frame_count * channels * bytes-per-channel), paContinue= c'è ancora audio in ingresso

    La funzione permette di prendere l'audio in input da microfono ed elaborarlo.

    '''
    # if status:
    #    print("Stato della callback: %i" % status)

    # variabili globali presentate anche nelle precedenti righe di codice
    global model, f, ind, buff, start_time, process, prev_chunk, step, incr, zeros_bytes, flagSample, prev_out, kick_nframes, hat_nframes, snare_nframes
    out_data = bytes()

    start_time = time.time() # tempo di inzio per misurare approssimativamente i tempi
    data_input = np.fromstring(in_data, dtype=np.int16)  # prendo in ingresso i byte
    data_input = data_input.astype(np.float32, order='C') / 32768.0  # converto audio a float32
    data = np.nan_to_num(data_input)  # controllo non vi siano nan
    indici = peakutils.indexes(data, thres=0.2, min_dist=1024, thres_abs=True)  # rilevo gli indici dei picchi

    if indici.size != 0:
        if indici[0] > 200:
            buff = data[indici[0]-200:indici[0] + 400]
        else:
            prec = 200 - indici[0]
            buff = np.concatenate((prev_chunk[-prec:], data[0:indici[0]+400]))
        mfcc = np.mean(librosa.feature.mfcc(y=buff, sr=RATE, n_mfcc=13).T, axis=0)  # su tale buffer, di 2048, calcolo le 13 mfcc
        X_test = scaler.transform(list(mfcc.reshape(1, -1)))  # normalizzo l'istanza di test
        label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
        print("label: ", label)  # stampo etichetta
        print("time: ", time.time() - start_time)
        #fn = 'samples/action-kick.wav' if label == 0 else 'samples/ec-hat002.wav' if label == 1 else 'samples/ec-sn004.wav'  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
        fn = kick if label == 0 else hat if label == 1 else snare  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
        # IL MULTIPROCESSING NON MI SEMBRA ABBASSARE MOLTO I TEMPI NELLA RIPRODUZIONE DEI CAMPIONI?!
        #process = multiprocessing.Process(target=playSound(label, fn))
        #process.start()
        #process.terminate()
        #print("time: ", time.time() - start_time)
        if label == 0:
            out_data = kick_bytes[step:incr]
            step = incr
            incr = incr + 2048
            prev_out = kick_bytes[step:incr]
        elif label == 1:
            out_data = hat_bytes[step:incr]
            step = incr
            incr = incr + 2048
            prev_out = hat_bytes[step:incr]
        else:
            out_data = snare_bytes[step:incr]
            step = incr
            incr = incr + 2048
            prev_out = snare_bytes[step:incr]
        print("out_data: ", out_data)
        prev_out = out_data
        flagSample = 1

    else:  # se non ho rilevato indici e non devo continuare il campione audio
        if flagSample == 1:
            out_data = prev_out
            print("out_data: ", out_data)
            print("passato")
        else:
            out_data = zeros_bytes

    prev_chunk = data
    return (out_data, pyaudio.paContinue)  # restituisco dati audio e flag per dire di continuare a restare in ascolto


# inizio la registrazione dal mic
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK,
                    stream_callback=callback)

# inizia lo stream
stream.start_stream()
# stampo l'avviso che sto registrando
print("recording...")

# aspetta che lo stream termini (15)
while stream.is_active():
    time.sleep(0.1)

# Fermo la registrazione
stream.stop_stream()
# Chiudo lo stream
stream.close()
# Chiudo istanza PyAudio
audio.terminate()

'''
# --------------------------------------- raccolgo un buffer di 2048 campioni
if indici.size != 0 and f == 0:  # se ho rilevato un picco e dunque il flag mi indica che è il primo blocco da 1024
    start_time = time.time()  # inizio a calcolare la latenza da qua (? - brutto, devo capire come calcolarla in modo esatto)
    buff = data[indici[0]:]  # memorizzo i campioni dall'indice del picco rilevato per tutto il blocco di 1024
    ind = indici[0]  # controllo che il picco sia unico all'interno di tale blocco - non dovrebbe servire ma per sicurezza tengo il controllo
    f = f + 1  # cambio flag per poi passare a memorizzare i campioni del blocco successivo
elif f == 1:  # primo blocco che segue quello contenente il picco
    buff = np.concatenate((buff, data))  # concateno ai campioni già memorizzati l'intero blocco corrente, al fine di avere infine un buffer di 2048
    f = f + 1  # cambio flag per poi passare a memorizzare i campioni del blocco successivo
elif f == 2:  # terzo ed ultimo blocco
    buff = np.concatenate((buff, data[0:ind]))  # ai campioni memorizzati finora, concateno un numero di campioni che mi consentano di avere un buffer completo di 2048
    # buff = librosa.util.normalize(buff)
    mfcc = np.mean(librosa.feature.mfcc(y=buff, sr=RATE, n_mfcc=13).T, axis=0)  # su tale buffer, di 2048, calcolo le 13 mfcc
    X_test = scaler.transform(list(mfcc.reshape(1, -1)))  # normalizzo l'istanza di test
    label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
    print("label: ", label)  # stampo etichetta
    fn = 'samples/action-kick.wav' if label == 0 else 'samples/ec-hat002.wav' if label == 1 else 'samples/ec-sn004.wav'  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare

    process = multiprocessing.Process(target=playSound(label))
    process.start()
    process.terminate()


    # with wave.open(fn) as fd:
    #    out_data = fd.readframes(1000000)
    #    print("out_data: ", out_data)
    #    print("in_data: ", in_data)

    # in_data = out_data
    # tempo = time.time() - start_time # rilevo il tempo impiegato per svolgere questi compiti
    # print("--- %s seconds ---" % (tempo)) # stampo tempo di latenza tra rilevamento picco e fin emissione del suono
    f = 0  # reimposto flag
'''
'''#------------------> utilizzando persino meno di 1024 campioni riesco a classificare correttamente polistirolo e pentolino
if indici.size != 0:
    mfcc = np.mean(librosa.feature.mfcc(y=data[indici[0]:], sr=RATE, n_mfcc=13).T, axis=0)  # su tale buffer, di 2048, calcolo le 13 mfcc
    X_test = scaler.transform(list(mfcc.reshape(1, -1)))  # normalizzo l'istanza di test
    label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
    print("label: ", label)  # stampo etichetta
    fn = 'samples/action-kick.wav' if label == 0 else 'samples/ec-hat002.wav' if label == 1 else 'samples/ec-sn004.wav'  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
    process = multiprocessing.Process(target=playSound(label))
    process.start()
    process.terminate()
'''
'''
if indici.size != 0:
    prec = 1024 - indici[0]
    buff = np.concatenate((prev_chunk[-prec:], data[0:indici[0]]))
    mfcc = np.mean(librosa.feature.mfcc(y=buff, sr=RATE, n_mfcc=13).T, axis=0)  # su tale buffer, di 2048, calcolo le 13 mfcc
    X_test = scaler.transform(list(mfcc.reshape(1, -1)))  # normalizzo l'istanza di test
    label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
    print("label: ", label)  # stampo etichetta
    fn = 'samples/action-kick.wav' if label == 0 else 'samples/ec-hat002.wav' if label == 1 else 'samples/ec-sn004.wav'  # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
    process = multiprocessing.Process(target=playSound(label))
    process.start()
    process.terminate()
'''