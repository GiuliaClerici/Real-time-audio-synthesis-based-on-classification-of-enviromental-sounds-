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
import time
from soundBrain import playSound

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
RECORD_SECONDS = 5  # tempo registrazione in secondi
WAVE_OUTPUT_FILENAME = "output.wav"  # nome con cui salverò il file audio
n_oggetti = 3
frames = []  # dichiaro var per frame
sgn = np.array([])  # var per segnale audio
dir = '.\\datasetRT\\'  # con linux poi i separatori saranno diversi
# listo i file .wav contenuti in quella cartella in una list filelist
# tr_sub_dirs è una lista di nomi delle sottocartelle di training, solo una per ora
tr_sub_dirs = ["training\\"]

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
        filename_wav = dir + tr_sub_dirs + "recordingRT-" + str(j) + ".wav"
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
        model = sklearn.mixture.GaussianMixture(n_components=n_oggetti)
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
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')
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
    model_filename = 'modelKmeansRT.sav' if flagAlg == 'K' else 'modelGmmRT.sav'
    try:
        model = joblib.load(model_filename)  # carico il modello ML
        scaler = joblib.load("scaler.sav")
        print("Caricamento del modello salvato come ",
              model_filename)  # stampo a video la conferma di aver caricato il modello
    except Exception as e:
        print("Errore nel trovare il file in cui è stato salvato il modello, chiamato: ",
              model_filename)  # qualora il modello non venga caricato, stampo un messaggio di errore
        print(e)  # stampo messaggio di errore

    # Startup - istanza pyaudio
    audio = pyaudio.PyAudio()
    prova = np.array([])
    c = 7 # 10
    buff = np.array([])

    def callback(in_data, frame_count, time_info, status):
        '''

        :param in_data: segnale audio in ingresso dal microfono
        :param frame_count: numero di frame
        :param time_info: informazioni sul tempo
        :param status: stato - PaCallbackFlags
        :return: dovrebbe essere (frame_count * channels * bytes-per-channel), paContinue= c'è ancora audio in ingresso

        La funzione permette di prendere l'audio in input da microfono ed elaborarlo.

        '''

        global model  # variabile globale contenente il modello ML
        global c
        global buff

        data_input = np.fromstring(in_data, dtype=np.int16)  # len(data) = 1024, type = np.ndarray - interpreta un buffer come un array 1-dimensionale
        data_input = data_input.astype(np.float32, order='C') / 32768.0  # converto audio a float32
        data = np.nan_to_num(data_input)
        indici = peakutils.indexes(data, thres=0.1, min_dist=10000, thres_abs=True)  # calcolo indici dei picchi con funzione predefinita
        if indici.size != 0 or c < 7:  # nel caso in cui non ci siano picchi
            #print("indici picchi:", indici)
            #print("c: ", c)
            start_time = time.time()
            if c == 7:
                buff = np.append(buff, data[int(indici[0]):])
                c = c - 1
                #print("indice picco: ", indici)
            elif c > 1:
                buff = np.append(buff, data)
                c = c - 1
            else:
                #start_time = time.time()
                dataN = librosa.util.normalize(buff)
                feats = np.mean(librosa.feature.mfcc(y=dataN, sr=RATE, n_mfcc=13).T, axis=0)
                X_test = scaler.transform(list(feats.reshape(1, -1)))
                label = model.predict(X_test)  # utilizzo mfcc da dare in input al modello ML
                tempo = time.time() - start_time
                print("label: ", label)  # stampo etichette
                print("--- %s seconds ---" % (tempo))
                playSound(label)
                buff = np.delete(buff, np.s_[0:])
                c = 7
        return (data, pyaudio.paContinue)  # restituisco dati audio e flag per dire di continuare a restare in ascolto


    # inizio la registrazione dal mic
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=False,
                        frames_per_buffer=CHUNK,
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
