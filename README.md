# Label Studio → YOLO & COCO Converter

This project automates converting a dataset downloaded from **Label Studio (COCO format with images)** into a directory layout and annotation structure compatible with **YOLOv5/Ultralytics** and **COCO (YOLOX)** training pipelines.

---

## 🚀 Features
- Filters the COCO JSON (`result.json`) to keep only selected categories.
- Splits dataset into Train / Validation / Test according to user-defined percentages.
- Exports both **YOLOv5-style** and **COCO-style** annotations.
- Copies and organizes corresponding image folders automatically.

---

## 📁 Input & Output Structure

### Input (from Label Studio)
```

labelstudio_download/
├─ result.json
└─ images/
├─ 000001.jpg
└─ ...

```

### Output (generated)
```

output_folder/
├─ yolo/
│  ├─ images/{train,val,test}/
│  └─ labels/{train,val,test}/
└─ coco/
├─ annotations/
│  ├─ train.json
│  ├─ valid.json
│  └─ test.json
├─ train2017/
├─ val2017/
└─ test2017/

````

---

## ⚙️ Installation

1. **Clone or copy** this repository locally.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows
````

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### 📦 requirements.txt

```txt
pylabel>=0.1.8
pandas>=1.4
numpy>=1.22
argparse
```

---

## 🧠 Usage

### Command-line Example

```bash
python main.py \
    -f /path/to/labelstudio_download \
    -o ./converted_dataset \
    -c "Phone hand" \
    -s 75 10 15
```

### Arguments

| Flag                   | Description                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `-f / --folder`        | Path to folder downloaded from Label Studio (must contain `result.json` and `images/`). |
| `-o / --output_folder` | Destination folder for new dataset splits.                                              |
| `-c / --categories`    | Space-separated list of class names to keep (inside quotes if more than one).           |
| `-s / --split`         | Three integers for Train / Val / Test percentages (e.g. `-s 75 10 15`).                 |

---

## 🧩 Example VS Code Debug Configuration

If you prefer using VS Code’s debugger, create or modify `.vscode/launch.json` as follows:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Split Label Studio Dataset",
            "type": "debugpy",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal",
            "args": [
                "-f=base_dir",
                "-c=Phone hand",
                "-o=./tmp",
                "-s", "75", "10", "15"
            ]
        }
    ]
}
```

Then:

1. Open VS Code in the project folder.
2. Select **Run → Start Debugging** or press **F5**.
3. The script will execute and create YOLO + COCO split datasets inside `./tmp/`.

---

## 🧹 Notes

* Export from Label Studio using **COCO format with images**.
* The script automatically remaps category IDs starting at 0.
* Temporary file `tmp.json` is created during filtering and deleted afterwards.
* Ensure your splits add up to 100.
* For multi-class filters, always quote the categories:
  e.g. `-c "Phone hand charger"`

---

## ✅ Example Result Summary

After running:

```
Number of images: 2000
Number of classes: 2
Classes: ['Phone', 'hand']
Class counts:
Phone: 1600
hand: 400
```

Output folders will be ready for YOLOv5 or COCO-compatible frameworks.

---

## 🧩 Extending

* Adjust naming of split folders (e.g., `train2017`) by editing the last part of `convert.py`.
* Add more exports (`PascalVOC`, `YOLOv8`, etc.) via `pylabel` if needed.


