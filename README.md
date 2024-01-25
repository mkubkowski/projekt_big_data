# projekt_big_data

## Mariusz - notatki 

### 1. Preprocessing

Ustawienia w punkcie 4: 8 workerów, G 2X, automatyczne dostosowanie liczby workerów. Robimy na całości. Powinno się zrobić w 3,5 godz.

Punkt 5: pamiętać o zrobieniu bucketu do wyników Atheny.

### Dodatkowe analizy
**Mariusz/lstm_clean.py** - model LSTM do wykrywania spoilerów, który testowałem u siebie lokalnie

Środowisko lokalne:

python 3.9.16

numpy 1.23.5

pandas 1.5.2

matplotlib 3.5.3

nltk 3.8.1

scikit-learn 1.2.0

gensim 4.3.0

tensorflow 2.10.0

Edytor: Spyder



Konfiguracja maszyny na AWS:

EC2

Ubuntu 22.0.4

Dysk 100 GB (10 GB nie starcza na Tensorflow)

r7i.large

Łączenie po SSH, edytor: Jupyter Notebook, plik **Mariusz/lstm_AWS.ipynb**, dane z bucketu S3

Środowisko:

python 3.10.12

numpy 1.23.5

pandas 1.5.2

matplotlib 3.5.3

nltk 3.8.1

scikit-learn 1.2.0

gensim 4.3.0

tensorflow 2.10.0

boto3 1.34.19

notebook 7.0.6

jupyter-server 2.12.5

seaborn 0.13.2
