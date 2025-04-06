import os
import time
import logging
import json
import torch

class Logger:
    def __init__(self, log_dir, name=None, log_level=logging.INFO):
        """
        初始化Logger
        Args:
            log_dir: 日志保存目录
            name: 日志名称
            log_level: 日志级别
        """
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_dir = log_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        if name is None:
            name = "train"
        
        self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # 配置logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 清除之前的handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 用于保存训练损失
        self.train_losses = {
            'render_loss': [],
            'size_loss': [],
            'mask_loss': [],
            'total_loss': []
        }
        
        # 记录开始时间
        self.start_time = time.time()
        
    def log_args(self, args):
        """记录训练参数"""
        self.logger.info("Training arguments:")
        args_dict = vars(args)
        self.logger.info(json.dumps(args_dict, indent=2, default=lambda x: str(x)))
        
        # 保存参数到JSON文件
        with open(os.path.join(self.log_dir, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=2, default=lambda x: str(x))
    
    def log_iteration(self, iteration, losses, learning_rate=None, additional_info=None):
        """记录每次迭代的损失"""
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
        """记录每个epoch的损失"""
        message = f"Epoch [{epoch}]"
        
        for k, v in epoch_losses.items():
            message += f", {k}: {v:.6f}"
        
        if duration is not None:
            message += f", duration: {duration:.2f}s"
            
        self.logger.info(message)
    
    def log_eval(self, metrics):
        """记录评估结果"""
        message = "Evaluation results:"
        
        for k, v in metrics.items():
            message += f" {k}: {v:.4f},"
            
        self.logger.info(message)
    
    def log_checkpoint(self, checkpoint_path):
        """记录检查点保存"""
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_losses(self):
        """保存训练损失到文件"""
        loss_file = os.path.join(self.log_dir, "training_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(self.train_losses, f, indent=2)
        self.logger.info(f"Training losses saved to {loss_file}")
    
    def log_training_complete(self):
        """记录训练完成"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # 保存损失数据
        self.save_losses()
