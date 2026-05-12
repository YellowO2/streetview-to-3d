# google-map-to-3d

## Requirements

- NVIDIA GPU with driver version ≥570 (to support CUDA 13.0)
- Python 3.12

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```
To verify:
```                                                                            
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); 
  print(torch.version.cuda)"              

  What you want to see:                                                                
  - 2.11.0 — correct torch version        
  - True — GPU is accessible                                                           
  - 13.0 — correct CUDA version     
```

**2. Configure model paths**

Edit `config.yaml` and set the absolute paths to the model files on your machine:

```yaml
models:
  sharp: /absolute/path/to/models/sharp_2572gikvuh.pt
  da3: /absolute/path/to/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```

**3. Run**

```bash
uvicorn main:app --reload
```

## Folder Structure
```
 main.py              # entry point — creates the FastAPI app, registers routes       
  config.py/yaml       # model paths and pipeline settings
                                                                                       
  routes/              # HTTP endpoints — what the frontend can call
    panorama.py        #   routes to do with fetching/downloading panoramas        
    splatting.py       #   routes relating to the 3DGS generation pipeline endpoints                      
                                                                                       
  services/            # functions used by routes
    download_street_panorama.py     #  download the panorama
    job_store.py       #   tracks job status (running/done/error) in memory            
                                                                                       
  ui/                  # frontend (vanilla JS, served as static files)                 
  images/              # downloaded panorama cache
  splats/              # generated .ply output files, one folder per job  
  ```