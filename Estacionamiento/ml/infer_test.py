import onnxruntime as ort
from PIL import Image
import numpy as np, sys

sess = ort.InferenceSession("../models/parking_free_vs_busy.onnx", providers=["CPUExecutionProvider"])

def predict(path):
    img = Image.open(path).convert("RGB").resize((224,224))
    x = np.asarray(img).astype("float32")/255.0
    x = x[None, ...]  # NHWC
    out = sess.run(None, {"input": x})[0].ravel()[0]  # sigmoid 0..1
    return ("LIBRE", float(out)) if out >= 0.5 else ("OCUPADO", float(out))

if __name__ == "__main__":
    img_path = sys.argv[1]
    print(predict(img_path))
