### Project contain
- python 3.5
- Tensorflow 1.2.0 keras
- Java
- Meka
### How to build
```
sudo docker build -t mlc .

```
### Test your MEKA file
```
sudo docker run -i -t  mlc
```
Example run file:
```
from skmultilearn.ext import Meka, download_meka
from skmultilearn.dataset import load_dataset
from sklearn.metrics import hamming_loss
import warnings
warnings.filterwarnings('ignore')

x_train, y_train, _, _ = load_dataset('scene', 'train')
x_test, y_test, _, _ = load_dataset('scene', 'test')
meka = Meka(
        meka_classifier = "meka.classifiers.multilabel.BR", # Binary Relevance
        weka_classifier = "weka.classifiers.bayes.NaiveBayesMultinomial", # with Naive Bayes single-label classifier
        meka_classpath = "/root/scikit_ml_learn_data/meka/meka-release-1.9.2/lib/", #obtained via download_meka
        java_command = '/usr/bin/java' # path to java executable
)
meka.fit(x_train, y_train)
predictions = meka.predict(x_test)
# res = meka._run("/workspace/scikit-multilearn/dataset/yeast/yeast-train.arff", "/workspace/scikit-multilearn/dataset/yeast/yeast-test.arff")
print(hamming_loss(y_test, predictions))
```
