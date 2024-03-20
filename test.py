import pyedflib
import numpy as np

if __name__ == "__main__":
    f = pyedflib.EdfReader("data/sleep-cassette/SC4082EP-Hypnogram.edf")
    print(f.getHeader())
    print(f.getSignalLabels())
    print(f.getSignalHeaders())
    print(f.getSex())
    print(f.readAnnotations())
    a, b, c = f.readAnnotations()
    print(list(a.astype(int)))
