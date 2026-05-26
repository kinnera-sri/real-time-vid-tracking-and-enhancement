# isap_planar
In Scene Ad Placement for Planar regions

# Directory organization
Currently all the code is put in the root folder. 
All the model files are kept in the models folder. 
The 3rd party models must be separately checked out and kept in their
respective folders


```
models/
├── lcnn
│   ├── config
│   ├── demo.py
│   ├── infer_lcnn.py
│   ├── lcnn
│   ├── model_inference.py
│   ├── post.py
│   ├── process.py
│   └── __pycache__
├── LightGlueGyrus
│   ├── hello_lggy.py
│   ├── lightglue_gy
│   └── robustpoint
└── third_party
    ├── Depth-Anything-V2 -> git clone from https://github.com/DepthAnything/Depth-Anything-V2
    ├── Depth-Anything-3 -> git clone from https://github.com/ByteDance-Seed/Depth-Anything-3
    ├── DSINE -> git clone from https://github.com/baegwangbin/DSINE
    └── TinySAM -> git clone from from https://github.com/xinghaochen/TinySAM
```
# Model checkpoints
All the model checkpoints are available from this link[https://drive.google.com/drive/folders/1IX072SXwss2d_iQR6TaPdxcmUWO1yYbZ?usp=sharing]. Please download and
unzip into assets/checkpoints folder


# Current flow
1. Download a video where the ad needs to be inserted. 
2. Run the scripts "scene\_analyzer", and "find\_apo" in sequence as shown below. 
```
python scene_analyzer.py --i ~/work/isr/target_videos/YoungSheldon_Locks_Himself_720P.mp4
python find_apo_v2.py 
```
These commands generate two json files in the staging\_dir folder. 

3. For blending the ad you need to use the code from this repo[https://github.com/shank-gyrus/ad_blending_flask_app]



