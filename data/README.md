## Data layout

This folder should contain dataset directories or symlinks that match the `images_path` entries in [`../configs/qwen3vl_decompose.yaml`](../configs/qwen3vl_decompose.yaml).

Large datasets are intentionally not committed to the repository. A typical setup is to keep the real data somewhere on shared storage and create symlinks here.

Example:

```bash
ln -s /path/to/miniimagenet data/miniimagenet
ln -s /path/to/CUB_200_2011 data/CUB_200_2011
```

## Download sources

Download the datasets from their original sources, then arrange or symlink them to match the structure below.

| Dataset key | Source | Notes |
| --- | --- | --- |
| `mini` | [mini-ImageNet on Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet?select=n01930112); [split files](https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/tree/master/filelists/miniImagenet) | Evaluation uses 20 classes. |
| `aircraft` | [FGVC-Aircraft on Kaggle](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft) | Data and split source; variant labels, 100 total classes. |
| `ucf` | [UCF101 official download page](https://www.crcv.ucf.edu/data/UCF101.php); [UTD debias split/method](https://utd-project.github.io/) | Download UCF101 from the official page. UTD is the debias method/split source. You must run `data/UCF101/extract_ucf_middle_frames.py` to extract one middle frame per video to `data/UCF101/UCF_frame/<video_id>.jpg` before evaluation. |
| `cub` | [CUB-200-2011 official dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/); [FRN split source](https://github.com/Tsingularity/FRN/tree/main) | Download data from the official CUB page; the repository link is only for the split. |
| `domain` | [DomainNet official download page](https://ai.bu.edu/M3SDA/#dataset); [DFRFS split source](https://github.com/chenghao-ch94/DFRFS) | Download data from the official DomainNet page; the repository link is only for the split. Domains must be `clipart`, `infograph`, `painting`, `quickdraw`, `real`, and `sketch`. |
| `industrial` | [Industrial Classification Data Set on Kaggle](https://www.kaggle.com/datasets/beschue/industrial-classification-data-set) | Use the test image folder under `IndustrialDataSet/Test_Dataset/images`. |
| `yoga` | [Yoga Pose Image Classification Dataset on Kaggle](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset) | Keep class folders under `data/yoga`. |
| `lego` | [Lego Brick Sorting - Image Recognition on Kaggle](https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition) | Keep brick-type folders directly under `data/lego`. |
| `arabicsign` | [RGB Arabic Alphabets Sign Language Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset) | Keep sign-class folders directly under `data/ArabicSign`. |
| `hieroglyph` | [Egyptian Hieroglyphs: GlyphDataset on Kaggle](https://www.kaggle.com/datasets/ahmedsamir1598/glyphdataset) | Keep `Manual/Preprocessed/<source_folder>/<image>.png`. |
| `dogs` | [Stanford Dogs Sanitised and Cropped 128x128 on Kaggle](https://www.kaggle.com/datasets/robbindg/stanford-dogs-sanitised-and-cropped-128-x-128?select=images) | Use the `images` folder as `data/dogs/images`. |
| `butterfly` | [Butterfly & Moths Image Classification 100 species on Kaggle](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species) | Evaluation paths use the `test/<CLASS>/<image>.jpg` split. |

## Expected structure

```text
data/
├── miniimagenet/
│   └── <class_name>/
│       └── <image>.JPEG
├── CUB_200_2011/
│   └── images/
│       └── <class_dir>/
│           └── <image>.jpg
├── dogs/
│   └── images/
│       └── <image>.jpg
├── ArabicSign/
│   └── <sign_class>/
│       └── <sign_class>_<index>.jpg
├── yoga/
│   └── <class_name>/
│       └── <image>.png
├── hieroglyphs/
│   └── Manual/
│       └── Preprocessed/
│           └── <source_folder>/
│               └── <image>.png
├── UCF101/
│   └── UCF_frame/
│       └── <video_id>.jpg
├── fgvc-aircraft-2013b/
│   └── data/
│       └── images/
│           └── <image>.jpg
├── DomainNet/
│   ├── clipart/
│   │   └── <class_name>/
│   │       └── <image>.jpg
│   ├── infograph/
│   │   └── <class_name>/
│   │       └── <image>.jpg
│   ├── painting/
│   │   └── <class_name>/
│   │       └── <image>.jpg
│   ├── quickdraw/
│   │   └── <class_name>/
│   │       └── <image>.jpg
│   ├── real/
│   │   └── <class_name>/
│   │       └── <image>.jpg
│   └── sketch/
│       └── <class_name>/
│           └── <image>.jpg
├── lego/
│   └── <lego_class>/
│       └── <image>.jpg
├── butterfly/
│   └── test/
│       └── <BUTTERFLY_OR_MOTH_CLASS>/
│           └── <image>.jpg
└── IndustrialDataSet/
    └── Test_Dataset/
        └── images/
            └── <image>.jpg
```

## Dataset-specific roots

| Dataset key | Expected root in this folder | Config `images_path` |
| --- | --- | --- |
| `mini` | `data/miniimagenet` | `data/miniimagenet` |
| `cub` | `data/CUB_200_2011` | `data/CUB_200_2011/images` |
| `dogs` | `data/dogs` | `data/dogs/images` |
| `arabicsign` | `data/ArabicSign` | `data/ArabicSign` |
| `yoga` | `data/yoga` | `data/yoga` |
| `hieroglyph` | `data/hieroglyphs` | `data/hieroglyphs` |
| `ucf` | `data/UCF101` | `data/UCF101/UCF_frame` |
| `aircraft` | `data/fgvc-aircraft-2013b` | `data/fgvc-aircraft-2013b/data/images` |
| `domain` | `data/DomainNet` | `data/DomainNet` |
| `lego` | `data/lego` | `data/lego` |
| `butterfly` | `data/butterfly` | `data/butterfly` |
| `industrial` | `data/IndustrialDataSet` | `data/IndustrialDataSet/Test_Dataset/images` |

## Dataset-specific examples

The episode logs already contain the relative file names expected by the reader. These examples show the exact pattern each less-standard dataset should follow.

### ArabicSign

ArabicSign uses one folder per sign class directly under `data/ArabicSign`.

```text
data/ArabicSign/
├── Ain/
│   ├── Ain_105.jpg
│   └── Ain_143.jpg
├── Ghain/
│   ├── Ghain_72.jpg
│   └── Ghain_186.jpg
├── Sad/
│   ├── Sad_195.jpg
│   └── Sad_257.jpg
└── ...
```

Episode paths look like:

```text
Ain/Ain_143.jpg
Sad/Sad_195.jpg
Ghain/Ghain_72.jpg
```

Because this dataset uses `prefixed_or_flat`, the reader resolves them as:

```text
data/ArabicSign/Ain/Ain_143.jpg
```

### Egyptian hieroglyphs

Hieroglyph images should be under `Manual/Preprocessed/` with the numeric source folder preserved.

```text
data/hieroglyphs/
└── Manual/
    └── Preprocessed/
        ├── 3/
        │   └── 030104_N5.png
        ├── 7/
        │   ├── 070136_N5.png
        │   └── 070394_E9.png
        ├── 20/
        │   └── 200367_U7.png
        ├── 22/
        │   ├── 220207_N31.png
        │   └── 220437_N31.png
        └── ...
```

Episode paths look like:

```text
Manual/Preprocessed/22/220207_N31.png
Manual/Preprocessed/7/070163_O49.png
Manual/Preprocessed/20/200367_U7.png
```

Because this dataset uses `prefixed_or_flat`, the reader resolves them as:

```text
data/hieroglyphs/Manual/Preprocessed/22/220207_N31.png
```

### DomainNet

DomainNet must contain all six domain folders:

```text
clipart/
infograph/
painting/
quickdraw/
real/
sketch/
```

Each domain folder should contain class folders, then images:

```text
data/DomainNet/
├── clipart/
│   ├── aircraft_carrier/
│   │   └── clipart_001_*.jpg
│   └── squirrel/
│       └── clipart_282_*.jpg
├── infograph/
│   ├── aircraft_carrier/
│   │   └── infograph_001_*.jpg
│   └── squirrel/
│       └── infograph_282_*.jpg
├── painting/
├── quickdraw/
├── real/
└── sketch/
```

Episode paths look like:

```text
painting/squirrel/painting_282_000151.jpg
sketch/skull/sketch_265_000392.jpg
real/aircraft_carrier/real_001_000225.jpg
infograph/hot_tub/infograph_152_000010.jpg
```

Because DomainNet uses `prefixed_or_flat`, the reader resolves them as:

```text
data/DomainNet/painting/squirrel/painting_282_000151.jpg
```

### UCF101

Download UCF101 videos from the [official UCF101 page](https://www.crcv.ucf.edu/data/UCF101.php). We use UTD as the debias method/split source; see the [UTD project page](https://utd-project.github.io/).

The evaluator does not read full videos. It expects one middle frame per video in a flat folder:

```text
data/UCF101/
├── UCF-101/
│   └── <action_class>/
│       └── <video_id>.avi
└── UCF_frame/
    └── <video_id>.jpg
```

You must use the provided helper script to extract middle frames before running UCF evaluation:

```bash
cd data/UCF101
python extract_ucf_middle_frames.py --videos_dir UCF-101 --output_dir UCF_frame
```

If your videos are stored elsewhere, pass that directory with `--videos_dir`. The output must remain flat:

```text
data/UCF101/UCF_frame/v_VolleyballSpiking_g07_c05.jpg
```

Episode paths store video IDs without the `.jpg` suffix:

```text
v_VolleyballSpiking_g07_c05
v_CuttingInKitchen_g04_c02
```

Because UCF uses `ucf_frame`, the reader appends `.jpg` and resolves them as:

```text
data/UCF101/UCF_frame/v_VolleyballSpiking_g07_c05.jpg
```

### Lego

Lego images use one folder per brick type directly under `data/lego`. Folder names use underscores, while class labels in the episode log may use spaces.

```text
data/lego/
├── Brick_1x1/
│   ├── 1_Brick_1x1_180708134114.jpg
│   └── 1_Brick_1x1_180708135921.jpg
├── Brick_1x4/
│   ├── 1_Brick_1x4_180708225312.jpg
│   └── 1_Brick_1x4_180708232941.jpg
├── Brick_2x4/
│   └── 1_Brick_2x4_180713182031.jpg
├── Plate_2x2/
│   └── 1_Plate_2x2_180714170536.jpg
└── ...
```

Episode paths look like:

```text
Brick_1x4/1_Brick_1x4_180708232941.jpg
Brick_1x1/1_Brick_1x1_180708135921.jpg
Plate_2x2/1_Plate_2x2_180714170536.jpg
```

Because Lego uses `prefixed_or_flat`, the reader resolves them as:

```text
data/lego/Brick_1x4/1_Brick_1x4_180708232941.jpg
```

### Butterfly

Butterfly and moth images are expected under the `test/` split. Class folders are uppercase names with spaces.

```text
data/butterfly/
└── test/
    ├── RED CRACKER/
    │   ├── 2.jpg
    │   └── 5.jpg
    ├── BLUE MORPHO/
    │   ├── 1.jpg
    │   └── 2.jpg
    ├── AN 88/
    │   ├── 1.jpg
    │   └── 5.jpg
    ├── DANAID EGGFLY/
    │   └── 1.jpg
    └── ...
```

Episode paths look like:

```text
test/RED CRACKER/2.jpg
test/BLUE MORPHO/2.jpg
test/AN 88/5.jpg
test/DANAID EGGFLY/1.jpg
```

Because Butterfly uses `prefixed_or_flat`, the reader resolves them as:

```text
data/butterfly/test/RED CRACKER/2.jpg
```

## How paths are resolved

The evaluator reads episode files from `episode_logs/<dataset>/`. Each episode contains class names and support/query file names. The reader turns those entries into full image paths using the dataset's `resolve_mode`.

- `class_subdir`: `images_path / class_name / image_name`
  - Used by `cub`.
- `mixed_class_or_prefixed`: if `image_name` already contains `/`, use `images_path / image_name`; otherwise use `images_path / class_name / image_name`.
  - Used by `mini` and `yoga`.
- `prefixed_or_flat`: `images_path / image_name`
  - Used by `dogs`, `arabicsign`, `hieroglyph`, `aircraft`, `domain`, `lego`, `butterfly`, and `industrial`.
  - For datasets such as DomainNet, the episode log stores paths like `<domain>/<class>/<image>.jpg`, so the same rule still works.
- `ucf_frame`: `images_path / "<video_id>.jpg"`
  - Used by `ucf`.
  - Episode logs store video IDs such as `v_VolleyballSpiking_g07_c05`; the reader appends `.jpg`.

If a dataset is stored differently, either rearrange/symlink it to match this layout or update the dataset entry in `configs/qwen3vl_decompose.yaml`.
