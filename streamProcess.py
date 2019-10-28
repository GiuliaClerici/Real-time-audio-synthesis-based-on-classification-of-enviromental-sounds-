import numpy as np
import librosa
import librosa.display
import glob
import os
import peakutils
import soundfile as sf
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot


def plot_peaks(x, indexes, algorithm=None, mph=None, mpd=None):
    '''
    :param x: segnale audio
    :param indexes: indici dei picchi
    :return: plot

    La funzione plotta i picchi rilevati sul segnale audio passato in ingresso.
    '''
    windows_ind = []
    windows_beg = []
    s_succ = 7000
    s_prev = 1000
    l = len(indexes)
    temp = 0
    for i in range(0, l):
        win = indexes[i] + s_succ
        pre = indexes[i] - s_prev
        windows_ind = np.append(windows_ind, win)
        windows_beg = np.append(windows_beg, pre)
    windows_ind = [int(x) for x in windows_ind]
    windows_ind = np.asarray(windows_ind)
    windows_beg = [int(x) for x in windows_beg]
    windows_beg = np.asarray(windows_beg)
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print('matplotlib is not available.', e)
        return
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b', lw=1)
    if indexes.size:
        label = 'peak'
        label = label + 's' if indexes.size > 1 else label
        ax.plot(indexes, x[indexes], '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (indexes.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    if windows_ind.size:
        label_w = 'end_window'
        label_w = label_w + 's' if windows_ind.size > 1 else label_w
        ax.plot(windows_ind, x[windows_ind], 'x', mfc=None, mec='y', mew=2, ms=8,
                label='%d %s' % (windows_ind.size, label_w))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    if windows_beg.size:
        label_b = 'beg_window'
        label_b = label_b + 's' if windows_beg.size > 1 else label_b
        ax.plot(windows_beg, x[windows_beg], 'o', mfc=None, mec='g', mew=2, ms=8,
                label='%d %s' % (windows_beg.size, label_b))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('%s (mph=%s, mpd=%s)' % (algorithm, mph, mpd))
    plt.show()


def peak_detection(fn):
    '''

    :param fn: file audio
    :return:

    La funzione si occupa di rilevare i picchi contenuti nel segnale audio.

    '''
    x = 0  # var che conterrà il mio segnale
    Fs = 0  # var per freq di campionamento
    thres = 0.06  # 0.27 # threshold
    x, Fs = sf.read(fn, dtype='int16')  # dtype='float64')  # leggo segnale audio, in x campioni audio del segnale e Fs freq campionamento
    if x.ndim > 1:  # se stereo
        x = np.mean(x, axis=1)  # converto in mono
    x = x.astype(np.float32, order='C') / 32768.0
    print("max val in x: ", np.amax(x))
    print("min val in x: ", np.amin(x))
    print("dim x: ", x.shape)
    xnew = librosa.util.normalize(x)
    print("max val in xnew: ", np.amax(xnew))
    print("min val in xnew: ", np.amin(xnew))
    print("dim xnew: ", xnew.shape)
    indexes = peakutils.indexes(xnew, thres=0.1, min_dist=5000, thres_abs=True)  # thres=0.1, min_dist=500
    # indexes = peakutils.indexes(x, thres=0.01, min_dist=5000, thres_abs=True) #thres=0.04, min_dist=500
    # print("peak utils: ", indexes)
    print("quanti peak utils alg: ", len(indexes))

    #plot_peaks(xnew, indexes) #<------------------- momentaneamente commentato

    return indexes, xnew, Fs


def feat_extraction(x, Fs, indexes, flagRT):
    '''

    :param x: segnale audio
    :param Fs: frequenza di campionamento
    :param indexes: array contenente le posizioni dei picchi nel segnale audio
    :return: mfcc del segnale audio

    La funzione si occupa di estrarre le features dal file audio.
    Nello specifico, calcola i 13 coefficienti mfcc per ogni finestra temporale relativa ad ogni picco rilevato.
    '''
    mfccs = []  # lista che conterrà mfcc
    cent = []
    feat = []  # ho sostituito mfccs con feat
    s_succ = 7000  # finestra temporale - di quanto altrimenti? vedi nel plot <-----------------------------------------------
    s_prev = 1000
    l = len(indexes)  # quanti indici di picchi
    # print("len: ", l)
    mfccs_mean = 0
    #print("flagRT: ", flagRT)
    for i in range(0, l):
        if flagRT == 0:
            #print("entra in flagrt=0")
            # if indexes[i] < temp: # se un indice rientra nella finestra temporale del precedente
            #    continue # salto all'iterazione successiva - ha senso? o calcolo ugualmente?<------------------------------
            win = indexes[i] + s_succ  # indice in cui termina la finestra temporale del picco in questione
            print("win: ", win)
            pre = indexes[i] - s_prev
            print("type x: ", x.size)
            if win > len(x):
                win = len(x) - 1
            if pre < 0:
                pre = 0
            print("prev: ", pre)
            a = x[pre:win]  # per ora non c'è overlap - segnale audio finestrato per il picco in questione
        else:
            #a = x
            a = x[indexes[i]:]
        #print("len finestra su cui calcolo mfcc: ", len(a))
        m = np.mean(librosa.feature.mfcc(y=a, sr=Fs, n_mfcc=13).T, axis=0)  # calcolo degli mfcc per la finestra - se voglio frame di 0.03 sec quindi 30ms allora n_ftt=0.03*sr e se voglio hop di 10ms quindi 0.01sec allora hop_length=0.01*sr
        # per modificare size dei frame inserire parametro n_fft = numero di campioni
        #print("mfcc: ", m)
        # print("mfcc shape: ", m.shape)
        # c = librosa.feature.spectral_centroid(y=a, sr=Fs) # calcolo spectral centroid
        # z = librosa.feature.zero_crossing_rate(a)
        # conc = np.concatenate((m, c), axis=None)
        # print("m: ", m)
        # print("c: ", c)
        # print(type(conc))
        # print("conc shape: ", conc.shape)
        # print("m+c: ", conc)
        # print("temp2: ", temp)
        if i == 0:  # se sto elaborando il primo picco
            feat.append(m)  # inserisco vettore di mfcc e spectral centroid del primo picco
            # cent.append(c)
        else:  # altrimenti
            feat = np.vstack((feat, m))  # inserisco gli mfcc dei picchi seguenti ognuno su una nuova riga
            # cent = np.vstack((cent, c))
    feat_np = np.asarray(feat)  # diventa un ndarray
    # cent_np = np.asarray(cent)
    # print("dentro feat_extraction che tipo è feat: ", type(feat))
    # print(feat.shape)
    #print("feat: ", feat)
    feat_mean = np.mean(feat_np, axis=0)  # RIPARTI DA QUI: è giusta la media mobile calcolata in questo modo?<-------
    # print("mfcc_np: ", mfccs_np)
    # print("mfcc: ", mfccs_mean)
    # print("mfcc len: ", mfccs_mean.shape)
    return mfccs_mean, feat_np  # , cent_np cosa voglio restituire? Gli mfcc mediati su tutto il segnale? Io non direi


# def pk(x, r):
#     y = np.zeros(x.shape)
#     rel = np.exp(-1 / r)
#     y[0] = np.abs(x[0])
#     for n, e in enumerate(np.abs(x[1:])):
#         y[n + 1] = np.maximum(e, y[n] * rel)
#     return y

def get_files_feat(parent_dir, sub_dirs, flagRT, file_ext="*.wav"):
    '''

    :param parent_dir: cartella genitore contenente cartelle figlie contenenti a loro volta i file audio
    :param sub_dirs: cartelle figlie contenenti al loro interno i file audio da elaborare
    :param file_ext: # estensione file audio - .wav
    :return: dataset di training

    La funzione analizza ogni segnale audio, estraendone le features.
    '''
    # features, labels = np.empty((0,193)), np.empty(0)
    # Preparo le variabili che conterranno le features e le etichette
    # La var per le features sarà una matrice in cui vi sarà una riga per ogni file e per ogni riga 160 colonne,
    # con 160 che è la somma del numero di features calcolate
    X_train = np.asarray([])
    features, labels = np.zeros((0, 160)), np.zeros(0)
    for label, sub_dir in enumerate(sub_dirs):
        print(label, sub_dir)
        # recupero il path dei file audio e per ognuno
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print("File che si sta analizzando:", fn)
            try:
                x = 0
                Fs = 0
                # per ogni file audio calcolo le features
                # x, Fs, mfccs, br = feature_extraction(fn)  # , spect_centr
                peaks, x, Fs = peak_detection(fn)
                print("numero picchi: ", len(peaks))
                mfccs_mean, feats = feat_extraction(x, Fs, peaks, flagRT)
                # print("File aperto e caricato")
            except Exception as e:
                # stampo un messaggio di errore qualora vi siano problemi nel caricamento del file audio e nel calcolo delle features
                print("Errore nel parsing dei file da cui estrarre le feature: ", fn)
                print(e)
                continue
            # if 'X_train' in locals():
            if X_train.size != 0:
                X_train = np.vstack((X_train, feats))  # tutte le mfcc collezionate
            else:
                X_train = feats
            for i in range(0, len(peaks)):
                labels = np.append(labels, fn.split('\\')[3].split('-')[1].split('.')[0])
            labels = np.array(labels, dtype=np.int)
            print("labels in get_feat_files: ", labels)
            print("X_train: ", X_train)
            print("X_train shape: ", X_train.shape)
    return X_train, labels
