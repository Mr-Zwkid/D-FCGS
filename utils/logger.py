import os
import time
import logging
import json
import torch

class Logger:
    def __init__(self, log_dir, name=None, log_level=logging.INFO):
        """
        Initialize Logger
        Args:
            log_dir: Directory to save logs
            name: Logger name
            log_level: Logging level
        """
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_dir = log_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        if name is None:
            name = "D-FCGS"
        
        self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear previous handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Set formatter
        formatter = logging.Formatter('\033[92m%(asctime)s - %(name)s - %(levelname)s\033[0m - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # For saving training losses
        self.train_losses = {
            'render_loss': [],
            'size_loss': [],
            'mask_loss': [],
            'total_loss': []
        }

        # For saving_general information
        self.general_info = {}
        
        # Record start time
        self.start_time = time.time()
        
    def log_args(self, args):
        """Log training arguments"""
        self.logger.info("Training arguments:")
        args_dict = vars(args)
        self.logger.info(json.dumps(args_dict, indent=2, default=lambda x: str(x)))
        
        # Save arguments to JSON file
        with open(os.path.join(self.log_dir, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=2, default=lambda x: str(x))
    
    def log_iteration(self, iteration, losses, learning_rate=None, additional_info=None):
        """Log losses for each iteration"""
        message = f"Iteration [{iteration}]"
        
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            message += f", {k}: {v:.6f}"
            if k in self.train_losses:
                self.train_losses[k].append(v)
        
        if learning_rate is not None:
            message += f", lr: {learning_rate:.6f}"
            
        if additional_info is not None:
            for k, v in additional_info.items():
                message += f", {k}: {v}"
        
        # self.logger.info(message)
    
    def log_epoch(self, epoch, epoch_losses, duration=None):
        """Log losses for each epoch"""
        message = f"Epoch [{epoch}]"
        
        for k, v in epoch_losses.items():
            message += f", {k}: {v:.6f}"
        
        if duration is not None:
            message += f", duration: {duration:.2f}s"
            
        self.logger.info(message)
    
    def log_eval(self, metrics):
        """Log evaluation results"""
        message = "Evaluation results:"
        
        for k, v in metrics.items():
            message += f" {k}: {v:.4f},"
            
        self.logger.info(message)
    
    def log_checkpoint(self, checkpoint_path):
        """Log checkpoint saving"""
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_losses(self):
        """Save training losses to file"""
        loss_file = os.path.join(self.log_dir, "training_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(self.train_losses, f, indent=2)
        self.logger.info(f"Training losses saved to {loss_file}")
    
    def log_training_complete(self):
        """Log training completion"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Save loss data
        self.save_losses()

    def log_info(self, message):
        """Log custom info messages"""
        self.logger.info(message)

    def create_list_for_logging(self, name):
        """Create a list for logging"""
        if name not in self.general_info:
            self.general_info[name] = []
    
    def add_to_list(self, name, value):
        """Add a value to a list for logging"""
        if name in self.general_info:
            self.general_info[name].append(value)
        else:
            self.logger.warning(f"List '{name}' not found in general info.")

    def save_general_info(self):
        """Save general information to a JSON file"""
        info_file = os.path.join(self.log_dir, "general_info.json")
        with open(info_file, 'w') as f:
            json.dump(self.general_info, f, indent=2)
        self.logger.info(f"General information saved to {info_file}")