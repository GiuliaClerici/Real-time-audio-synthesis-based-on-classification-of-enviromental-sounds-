import pyaudio  # Soundcard audio I/O access library
import wave  # Python 3 module for reading / writing simple .wav files
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.signal as scisi
import scipy
import peakutils
import librosa
import sklearn
from sklearn.externals import joblib
#   from processAudio import get_files_feat, feat_extraction
from sklearn import metrics
from streamProcess import get_files_feat, feat_extraction, plot_peaks
import glob
import os
# from featureExtraction import feature_extraction
import soundfile as sf
import peakutils
import csv
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot
from sklearn.preprocessing import StandardScaler
from soundBrain import playSound
from pyo import *
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
import playsound as ps

# -------------------------------------sia Training che Test------------------------------------------------------------
# flag per decidere modello: K = Kmeans, G = GMM
flagAlg = 'K'
# Ulteriore flag: 0 = fai training, 1 = fase di test, non eseguire training ma carica modello
flagTrain = 1

# Setup
FORMAT = pyaudio.paInt16  # formato dati (opp. FORMAT = pyaudio.paInt16) #<--------------------------------!!!!!!
CHANNELS = 1  # numero canali
RATE = 44100  # Sample Rate
CHUNK = 1024  # Block Size
CHUNK_TEST = 1024 * 14  # = 15.360 quindi 0,32ms  14.336
RECORD_SECONDS = 60  # tempo registrazione in secondi
WAVE_OUTPUT_FILENAME = "output.wav"  # nome con cui salverò il file audio
n_oggetti = 3
frames = []  # dichiaro var per frame
sgn = np.array([])  # var per segnale audio
dir = '.\\datasetRT\\'  # con linux poi i separatori saranno diversi
# listo i file .wav contenuti in quella cartella in una list filelist
# tr_sub_dirs è una lista di nomi delle sottocartelle di training, solo una per ora
tr_sub_dirs = ["training\\"] #["training\\"]

# Startup - istanza pyaudio
audio = pyaudio.PyAudio()
print(audio.get_default_input_device_info())
print(audio.get_device_info_by_index(1))  # cerco index del dispositivo, per ora mic del pc

if flagTrain == 0:
    '''
    # inizio la registrazione dal mic
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    for j in range(0,n_oggetti):  # per ognuna delle quattro categorie di suoni, registro un file audio che mi servirà per il training
        # stampo l'avviso che sto registrando
        print("recording...")
        # inizia lo stream di registrazione
        stream.start_stream()

        # Registro per RECORD_SECONDS secondi
        print(int(RATE / CHUNK * RECORD_SECONDS))
        c = 0  # var di ciclo
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  # per i secondi di registrazione prestabili
            c = c + 1
            # leggo i chunk di dati in ingresso
            chunkdata = stream.read(CHUNK)
            print("chunk len: ", len(chunkdata))
            # converto la stringa di byte in un array numpy
            data = np.fromstring(chunkdata, dtype=np.int16)  # dtype=np.float32 è giusto che converta in float32?<-------------------------
            # opp. = np.frombuffer(chunkdata, dtype=np.float32)
            # li memorizzo nella var frames
            frames.append(data)  # me li registra tutti alla fine, con shape (215, 512) - in pratica memorizza un chunk in ogni posizione
        # avviso che la registrazione è finita, è terminato il tempo
        print("finished recording")
        print("c: ", c)

        # prova:
        #print("size data: ", len(data))
        print("size frame: ", len(frames))

        # Fermo la registrazione
        stream.stop_stream()

        # trasformo frames in un array numpy tenendo presente il formato dei dati
        #s = np.asarray(frames, dtype=np.float32) #<--------------------------------------------------------------

        #audio_sign =np.asarray(frames)
        #print("audio_sign: ", audio_sign)
        #print("len audio: ", audio_sign.shape)
        #s = audio_sign.astype(np.float32, order='C') / 32768.0
        # stampo a video i dati
        #s = np.nan_to_num(s)

        s = np.asarray(frames) 
        print("s: ", s)
        print(s.shape)
        print(len(s[214]))


        # apro il file WAVE_OUTPUT_FILENAME in modalità Write only in modo tale da creare il file audio contenente la registrazione
        filename_wav = dir + tr_sub_dirs[0] + "recordingRT-" + str(j) + ".wav"
        waveFile = wave.open(filename_wav, 'wb')
        # imposto il numero di canali
        waveFile.setnchannels(CHANNELS)
        # imposto formato dati per ampiezza campioni
        waveFile.setsampwidth(audio.get_sample_size(FORMAT)) # setsampwidth(audio.get_sample_size(FORMAT)) #<-------- NO!!!
        # imposto frequenza di campionamento
        waveFile.setframerate(RATE)
        # scrivo i dati di frames nel file
        waveFile.writeframes(b''.join(s))
        # chiudo il file
        waveFile.close()

        # numpydata = np.hstack(a)
        # plotto il segnale
        plt.plot(s)
        # s = s[5]
        # print(s)
        # peaks = peakutils.peak.indexes(s, thres=0.8)
        # print(peaks)
        # print(s[peaks])
        # plt.plot(s)
        # plt.plot(peaks, "o")
        plt.title("Prova")
        plt.show()
        

        #sgn = np.absolute(s)
        #plt.specgram(sgn.flatten(), Fs=44100)
        #plt.title("Spettrogramma")
        #plt.show()
        frames.clear()
    
    # Chiudo lo stream
    stream.close()
    # Chiudo istanza PyAudio
    audio.terminate()
    '''
    # chiamo la funzione get_files_feat contenuta nel file processAudio - in cui recupero i file, rilevo i picchi e calcolo gli mfcc
    X_train, y_train = get_files_feat(dir, tr_sub_dirs, 0)
    print("X_train: ", X_train)
    scaler = StandardScaler().fit(list(X_train))
    X_train = scaler.transform(list(X_train))
    scaler_filename = "scalerRT.sav"
    joblib.dump(scaler, scaler_filename)

    if flagAlg == 'K':
        # training del modello Kmeans
        # utilizzo algoritmo di ML Kmeans con 3 cluster, seme = 0, 10 iterazioni con diversi centroidi
        model = sklearn.cluster.KMeans(n_clusters=n_oggetti, random_state=0, n_init=10)
        # addestro sulle feature reperite dai file audio del dataset di training
        model.fit(X_train)
        # memorizzo le etichette assegnate ai punti di training
        labels = model.labels_
    else:
        # algoritmo modello di misture di Gaussiane a 3 componenti
        model = GaussianMixture(n_components=n_oggetti)
        model.fit(X_train)
        labels = model.predict(X_train)

    # print(labels)

    outputFile = 'csv\\labelsKmeans_trainRT.csv' if flagAlg == 'K' else 'csv\\labelsGMM_trainRT.csv'
    # creo file/riscrivo uno già esistente
    file = open(outputFile, 'w+')  # make file/over write existing file
    # salvo il file
    np.savetxt(file, labels, delimiter=",")  # save MFCCs as .csv
    # chiudo il file
    file.close()  # close file

    if flagAlg == 'K':
        # per scrupolo e curiosità, guardo che etichette vengono predette per i punti del training set
        y_kmeans = model.predict(
            X_train)  # non è sempre labels della riga 39??<---------------------------------------------
        # mostro in un grafico i diversi punti
        LABEL_COLOR_MAP = {0: 'r',
                           1: 'y',
                           2: 'b'
        }
        label_color = [LABEL_COLOR_MAP[l] for l in labels]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=label_color, s=50, cmap='viridis')
        # seleziono i centroidi dei cluster
        centers = model.cluster_centers_
    else:
        # per GMM:
        y_gmm = model.predict(
            X_train)  # non è sempre labels della riga 44??<------------------------------------------------
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_gmm, s=50, cmap='viridis')
        centers = np.empty(shape=(model.n_components, X_train.shape[1]))
        for i in range(model.n_components):
            density = scipy.stats.multivariate_normal(cov=model.covariances_[i], mean=model.means_[i]).logpdf(X_train)
            centers[i, :] = X_train[np.argmax(density)]
    # mostro i centroidi nel grafico
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # mostro il plot
    plt.show()

    # plotto anche un grafico 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    N = 449
    colors = np.random.rand(N)
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], marker='o', c=labels)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=20, marker='x', alpha=1)
    plt.show()

    # nome del file in cui salverò il modello
    model_filename = 'modelKmeansRT.sav' if flagAlg == 'K' else 'modelGmmRT.sav'
    # salvo il modello nel file model_filename
    joblib.dump(model, model_filename)
    # se volessi caricare il modello
    # loaded_model = joblib.load(model_filename)
    print("y_train: ", y_train)
    print("labels: ", labels)
    print("Accuracy sul training set:", metrics.adjusted_rand_score(y_train, labels))
    print("Fine training.")

else:
    # carico il modello di ML addestrato e lo scaler per normalizzare le istanze di test
    model_filename = 'modelKmeansRT.sav' if flagAlg == 'K' else 'modelGmmRT.sav'
    try:
        model = joblib.load(model_filename)  # carico il modello ML
        scaler = joblib.load("scaler.sav")
        print("Caricamento del modello salvato come ", model_filename)  # stampo a video la conferma di aver caricato il modello
    except Exception as e:
        print("Errore nel trovare il file in cui è stato salvato il modello, chiamato: ", model_filename)  # qualora il modello non venga caricato, stampo un messaggio di errore
        print(e)  # stampo messaggio di errore

    # Startup - istanza pyaudio
    audio = pyaudio.PyAudio()
    # inizializzo le variabili globali che utilizzo all'interno della funzione di callback
    f = 0 # flag per indicare quanti blocchi/chunk ancora dovrò prendere e per far sì che siano di 2048
    ind = 0 # variabile che conterrà l'indice del picco rilevato e che mi consentirà di selezionare esattamente 2048 campioni
    buff = np.zeros(2048) # buffer che conterrà 2048 campioni audio, a partire dal picco rilevato, su cui calcolo le mfcc e predico un'etichetta
    start_time = 0 # variabile per calcolare quanto tempo viene impiegato dal rilevamento del picco al rilascio del suono di output triggerato dall'etichettatura

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
        global model
        global f
        global ind
        global buff
        global start_time

        data_input = np.fromstring(in_data, dtype=np.int16)  # prendo in ingresso i byte
        data_input = data_input.astype(np.float32, order='C') / 32768.0  # converto audio a float32
        data = np.nan_to_num(data_input) # controllo non vi siano nan
        indici = peakutils.indexes(data, thres=0.2, min_dist=1024, thres_abs=True)  # rilevo gli indici dei picchi

        #--------------------------------------- buffer di 2048 campioni
        if indici.size != 0 and f == 0: # se ho rilevato un picco e dunque il flag mi indica che è il primo blocco da 1024
            start_time = time.time() # inizio a calcolare la latenza da qua (? - brutto, devo capire come calcolarla in modo esatto)
            buff = data[indici[0]:] # memorizzo i campioni dall'indice del picco rilevato per tutto il blocco di 1024
            ind = indici[0] # controllo che il picco sia unico all'interno di tale blocco - non dovrebbe servire ma per sicurezza tengo il controllo
            f = f + 1 # cambio flag per poi passare a memorizzare i campioni del blocco successivo
        elif f == 1: # primo blocco che segue quello contenente il picco
            b = np.concatenate((buff, data)) # concateno ai campioni già memorizzati l'intero blocco corrente, al fine di avere infine un buffer di 2048
            f = f + 1 # cambio flag per poi passare a memorizzare i campioni del blocco successivo
        elif f == 2: # terzo ed ultimo blocco
            buff = np.concatenate((buff, data[0:ind])) # ai campioni memorizzati finora, concateno un numero di campioni che mi consentano di avere un buffer completo di 2048
            mfcc = np.mean(librosa.feature.mfcc(y=buff, sr=RATE, n_mfcc=13).T, axis=0) # su tale buffer, di 2048, calcolo le 13 mfcc
            X_test = scaler.transform(list(mfcc.reshape(1, -1))) # normalizzo l'istanza di test
            label = model.predict(X_test)  # predico un'etichetta per il picco rilevato => buffer
            print("labels: ", label)  # stampo etichetta
            fn = 'samples/Drum_snare_shortened.mp3' if label == 0 else 'samples/Drum_kick_shortened.mp3' if label == 1 else 'samples/Drum_crash_shortened.mp3' # in base all'etichetta ottenuta seleziono il nome del file audio da caricare
            ps.playsound(fn) # triggero il suono in base all'etichetta, triggerando uno specifico campione di batteria
            tempo = time.time() - start_time # rilevo il tempo impiegato per svolgere questi compiti
            print("--- %s seconds ---" % (tempo)) # stampo tempo di latenza tra rilevamento picco e fin emissione del suono
            f = 0 # reimposto flag
        return (data, pyaudio.paContinue)  # restituisco dati audio e flag per dire di continuare a restare in ascolto


    # inizio la registrazione dal mic
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=False, frames_per_buffer=CHUNK,
                        stream_callback=callback)
    # stampo l'avviso che sto registrando
    print("recording...")
    # inizia lo stream
    stream.start_stream()

    # aspetta che lo stream termini (15)
    while stream.is_active():
        time.sleep(15)

    # Fermo la registrazione
    stream.stop_stream()
    # Chiudo lo stream
    stream.close()
    # Chiudo istanza PyAudio
    audio.terminate()

'''
        #----------------------------------------- buffer di solo 1024 campioni 
        if indici.size != 0:  # nel caso in cui non ci siano picchi
            print("indici: ", indici)
            print("max: ", data.max())
            start_time = time.time()
            mfccs_np, feats = feat_extraction(data, RATE, indici, 0)  # mfccs = np.asarray(feat_extraction(data, RATE, indici)) - calcolo mfcc del segnale audio
            X_test = scaler.transform(list(feats))
            labels = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
            tempo = time.time() - start_time
            print("labels: ", labels)  # stampo etichette
            print("--- %s seconds ---" % (tempo))
'''
'''
        #----------------------------------------- buffer esattamente da 1024 in due cicli 
        if indici.size != 0 and f == 0:
            start_time = time.time()
            #print("prima finestra con indice: ", indici)
            b = data[indici[0]:]
            len = np.size(b)  # se è un numpy array, altrimenti calcola la size in altro modo
            #print("len: ", len)
            f = f + 1
        elif f == 1:
            p = 1024 - len
            #print("p: ", p)
            buff = np.concatenate((b, data[0:p]))
            #buff = librosa.util.normalize(buff)
            indice = 1024 - len
            mfccs_np, feats = feat_extraction(buff, RATE, indice, flagRT=1)
            X_test = scaler.transform(list(feats.reshape(1, -1)))
            labels = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
            tempo = time.time() - start_time
            print("labels: ", labels)  # stampo etichette
            print("--- %s seconds ---" % (tempo))
            f = 0

'''
'''
        # ----------------------------------------- buffer in cui prendo 1000 campioni che precedono il picco che costituiscono l'attacco
        if indici.size != 0:
            start_time = time.time()
            inizio = indici[0] - 200
            i = inizio if inizio >= 0 else 0
            fine = indici[0] + 1000
            f = fine if fine <= 1024 else 1024 #  DA SISTEMARE PERCHÈ HO BISOGNO DI DUE BUFFER
            #print("indice: ", indici[0])
            #print("i: ", i)
            #print("signal: ", data[i:indici[0]])
            #print("shape: ", data[i:indici[0]].shape)
            feats = np.mean(librosa.feature.mfcc(y=data[i:f], sr=RATE, n_mfcc=13).T, axis=0)
            X_test = scaler.transform(list(feats.reshape(1, -1)))
            label = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
            tempo = time.time() - start_time
            print("label: ", label)  # stampo etichette
            print("--- %s seconds ---" % (tempo))
'''
'''
        #------------------------------------------ buffer di 400 circa, con 200 campioni prima del picco e 200 dopo 
        if indici.size != 0:
            start_time = time.time()
            inizio = indici[0] - 200
            fine = indici[0] + 200
            i = inizio if inizio >= 0 else 0
            f = fine if fine <= 1024 else 1024
            mfccs_np, feats = feat_extraction(data[i:f], RATE, indici, flagRT=1)
            X_test = scaler.transform(list(feats.reshape(1, -1)))
            labels = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
            tempo = time.time() - start_time
            print("labels: ", labels)  # stampo etichette
            print("--- %s seconds ---" % (tempo))
            f = 0

'''
'''
        #----------------------------------------- buffer di 2600 campioni (valore scelto da osservazioni empiriche - spettri) 
        if indici.size != 0 and f == 0:
            start_time = time.time()
            # print("prima finestra con indice: ", indici)
            b = data[indici[0]:]
            len = np.size(b)  # se è un numpy array, altrimenti calcola la size in altro modo
            # print("len: ", len)
            f = f + 1
        elif f == 1:
            b = np.concatenate((b, data))
            f = f + 1
        elif f == 2:
            p = 2600 - len
            # print("p: ", p)
            buff = np.concatenate((b, data[0:p]))
            # buff = librosa.util.normalize(buff)
            indice = 2600 - len
            mfccs_np, feats = feat_extraction(buff, RATE, indice, flagRT=1)
            X_test = scaler.transform(list(feats.reshape(1, -1)))
            labels = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
            tempo = time.time() - start_time
            print("labels: ", labels)  # stampo etichette
            print("--- %s seconds ---" % (tempo))
            f = 0
'''
'''
        if indici.size != 0 or c < 6:  # nel caso in cui non ci siano picchi
            if c == 6:
                start_time = time.time()
                buff = np.append(buff, data[int(indici[0]):])
                ind = indici[0]
                c = c - 1
                #print("indice picco: ", indici)
            elif c > 1:
                buff = np.append(buff, data)
                c = c - 1
            else:
                #start_time = time.time()
                buff = np.append(buff, data[0:ind])
                dataN = librosa.util.normalize(buff)
                print("shape buff: ", dataN.shape)
                feats = np.mean(librosa.feature.mfcc(y=dataN, sr=RATE, n_mfcc=13).T, axis=0)
                X_test = scaler.transform(list(feats.reshape(1, -1)))
                label = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
                tempo = time.time() - start_time
                print("label: ", label)  # stampo etichette
                print("--- %s seconds ---" % (tempo))
                #playSound(label) #<------------------------------------- suono!
                #print("--- %s seconds after sound ---" % (time.time() - start_time))
                buff = np.delete(buff, np.s_[0:])
                ind = 0
                c = 6
'''