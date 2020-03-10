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
#   from processAudio import get_files_feat, feat_extraction
import joblib
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
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
import playsound as ps
import sounddevice as sd
import soundfile as sf

print(sklearn.__version__)
# -------------------------------------sia Training che Test------------------------------------------------------------
# flag per decidere modello: K = Kmeans, G = GMM
flagAlg = 'K'
duration = 60  # tempo registrazione in secondi
sd.default.channels = 1
fs = 44100
sd.default.samplerate = fs
n_oggetti = 3
dir = '.\\datasetRT\\'  # con linux poi i separatori saranno diversi
# tr_sub_dirs è una lista di nomi delle sottocartelle di training, solo una per ora
tr_sub_dirs = ["training\\"]  # ["training\\"]

for j in range(0,n_oggetti):  # per ognuna delle quattro categorie di suoni, registro un file audio che mi servirà per il training
    # stampo l'avviso che sto registrando
    print("recording...")
    # inizia lo stream di registrazione

    myrecording = sd.rec(int(duration * fs), dtype='float64')
    sd.wait()
    print("myrec: ", myrecording.size)
    filename_wav = dir + tr_sub_dirs[0] + "recordingRT-" + str(j) + ".wav"
    sf.write(filename_wav, myrecording, fs)

    # avviso che la registrazione è finita, è terminato il tempo
    print("finished recording")

    # plotto il segnale
    plt.plot(myrecording)
    # s = s[5]
    # print(s)
    # peaks = peakutils.peak.indexes(s, thres=0.8)
    # print(peaks)
    # print(s[peaks])
    # plt.plot(s)
    # plt.plot(peaks, "o")
    plt.title("Prova")
    plt.show()

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
model_filename = 'modelKmeansRTAttack600Mano.sav' if flagAlg == 'K' else 'modelGmmRT.sav'
# salvo il modello nel file model_filename
joblib.dump(model, model_filename)
# se volessi caricare il modello
# loaded_model = joblib.load(model_filename)
print("y_train: ", y_train)
print("labels: ", labels)
print("Accuracy sul training set:", metrics.adjusted_rand_score(y_train, labels))
print("Fine training.")