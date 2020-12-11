import nltk
print(nltk.__version__)

# nltk.set_proxy('http://192.168.0.134:8118', ('USERNAME', 'PASSWORD'))
nltk.set_proxy('http://192.168.0.134:8118')
nltk.download('cmudict')
