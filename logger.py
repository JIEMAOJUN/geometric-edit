import json
from pathlib import Path
from datetime import datetime


class Logger:
    
    def __init__(self, exp_name: str, save_dir: str = 'results'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(save_dir) / f"{exp_name}_{timestamp}"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_file = self.log_dir / 'log.txt'
        self.metrics_file = self.log_dir / 'metrics.json'
        
        self.log(f"Experiment: {exp_name}")
        self.log(f"Log directory: {self.log_dir}")
    
    def log(self, message: str):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
    
    def save_metrics(self, metrics: dict):
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.log(f"Metrics saved to {self.metrics_file}")
    
    def get_log_dir(self) -> Path:
        return self.log_dir
