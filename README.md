# SE4050-Deep-Learning-Lab-Sheet-3

## Lab 4: Application of Convolutional Neural Networks – Part 1
### Face Recognition (Siamese Network / FaceNet)

---

## Overview
This lab uses a pretrained Siamese network (Inception-based FaceNet model) to perform
**face verification** and **face recognition** on the "Happy House" dataset, then extends
it with two new images of an individual not originally in the database.

---

## Files in this submission

| File | Description |
|---|---|
| `Face_Recognition_for_the_Happy_House_v2.ipynb` | Modified notebook with the new identity added to the database and verification/recognition run using the new images |
| `me1.jpg` | Image used to add the new identity to the database |
| `me2.jpg` | A different image of the same individual, used to test verification and recognition |
| `Lab4_FaceRecognition.docx` | Word file containing the two new images and screenshots of the verification/recognition outputs, including `output[2]` |
| `README.md` | This file |

---

## What was done

1. **Environment setup**
   - Ran the notebook locally via Jupyter Notebook inside an Anaconda environment (`se4050`)
   - Installed missing dependency: `opencv-python` (for `cv2`, used by `fr_utils.py`)

2. **Added a new identity to the database**
   ```python
   database["chathu"] = img_to_encoding("images/me1.jpg", FRmodel)
   ```

3. **Face verification** — tested with the second image against the new identity:
   ```python
   verify("images/me2.jpg", "chathu", database, FRmodel)
   ```

4. **Face recognition** — tested with the second image against the full database:
   ```python
   output = who_is_it("images/me2.jpg", database, FRmodel)
   ```

5. **Distance dictionary** — printed `output[2]`, the L2 distances between the target
   image encoding and every embedding in the database.

---

## Notes / Known limitations

- The pretrained FaceNet model was trained on the original "Happy House" dataset members
  (Younes, Bertrand, etc.), not on the new identity added here. Because of this, the
  recognition result may not always correctly match the new identity (a lower distance
  to an existing database member is possible) — this is expected behavior for a small,
  unfine-tuned database and does not indicate a code error.
- Images were resized to 96×96 pixels (the model's required input shape) before encoding.
- No GPU was used (TensorFlow ≥2.11 does not support native GPU on Windows) — inference
  ran on CPU.

---

## How to reproduce

1. Open `Face_Recognition_for_the_Happy_House_v2.ipynb` in Jupyter (with `fr_utils.py`,
   `inception_blocks_v2.py`, the `weights` folder, and an `images` folder containing
   `me1.jpg` and `me2.jpg`, all in the same working directory).
2. Run all cells top to bottom.
3. Verification and recognition results will print inline, matching the screenshots
   included in `Lab4_FaceRecognition.docx`.
