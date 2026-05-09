  Critiques and improves AOC-IDS (Zhang et al., IEEE INFOCOM 2024). Proposes GS-CRC loss with K-component GMM detection, EMA centroid tracking,
   and KL-drift triggered retraining.
                                                                                                                                               
  ## Datasets                                               
  - NSL-KDD: `AOC-IDS/NSL_pre_data/`
  - UNSW-NB15: `AOC-IDS/UNSW_pre_data/`                                                                                                        
   
  ## Run                                                                                                                                       
  ```bash                                                   
  # Baseline
  python -u AOC-IDS/online_training.py --dataset unsw --epochs 800
                                                                                                                                               
  # GS-CRC v2                                                                                                                                  
  python -u AOC-IDS/online_training_improved_v2.py --dataset unsw --epochs 800 --eps_drift 0                                                   
                                                                                                                                               
  Paper                                                     
                                                                                                                                               
  See GS_CRC_Paper.pdf.   
