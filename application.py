from flask import Flask, request
import librosa
import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
import joblib
from scipy import stats
import io, base64

application = Flask(__name__)

# model = tf.keras.models.load_model("models/model_10.h5", compile=False)

interpreter = tf.lite.Interpreter(model_path="models/model_v2.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
pipe, enc = joblib.load("models/pipe_10.joblib"), joblib.load("models/enc_10.jobilb")


def columns():
  feature_sizes = dict(
    chroma_cens=12,
    tonnetz=6,
    mfcc=20,
    zcr=1,
    spectral_centroid=1,
    spectral_contrast=7,
  )
  moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

  columns = []
  for name, size in feature_sizes.items():
    for moment in moments:
      it = ((name, moment, '{:02d}'.format(i + 1)) for i in range(size))
      columns.extend(it)

  names = ('feature', 'statistics', 'number')
  columns = pd.MultiIndex.from_tuples(columns, names=names)

  return columns.sort_values()

def compute_features(x, sr):
    features = pd.Series(index=columns(), dtype=np.float32)
    warnings.filterwarnings('error', module='librosa')
        
    def feature_stats(name, values):
        features.loc[(name, 'mean')] = np.mean(values, axis=1)
        features.loc[(name, 'std')] = np.std(values, axis=1)
        features.loc[(name, 'skew')] = stats.skew(values, axis=1)
        features.loc[(name, 'kurtosis')] = stats.kurtosis(values, axis=1)
        features.loc[(name, 'median')] = np.median(values, axis=1)
        features.loc[(name, 'min')] = np.min(values, axis=1)
        features.loc[(name, 'max')] = np.max(values, axis=1)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)
    cqt = None
    
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    x = None

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    stft = None

    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)
    
    f = None
    return (features)

def find_genre(y, sr):
    features = compute_features(y, sr)
    columns = ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr', 'tonnetz']
    features = features.loc[columns]
    transposed_df = pd.DataFrame(features.values.reshape(1, -1),
                                columns=features.index)
    features = pipe.transform(transposed_df)
    features = np.array(features, dtype=np.float32)

    input_shape = input_details[0]['index']
    interpreter.set_tensor(input_shape, features)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds = np.argsort(output_data.reshape(-1))
    features = None
    transposed_df = None
    return enc.inverse_transform(preds)[::-1]

@application.route('/', methods=['POST'])
def index():
    decoded = request.files['file']
    content = decoded.read()
    file = io.BytesIO(content)
    file.seek(0)
    y, sr = librosa.load(file)
    print(type(y), type(sr))
    genres = find_genre(y, sr).tolist()
    print(genres)
    y = None
    return f"{genres}"

application.run(host='0.0.0.0', port=80, threaded=True, debug=True)
