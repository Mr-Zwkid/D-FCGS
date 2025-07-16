import GPUtil  

def get_least_used_gpu():  
    # 获取可用的GPU列表  
    gpus = GPUtil.getGPUs()  
    
    # 初始化变量来跟踪最少使用的GPU  
    least_used_gpu_id = None  
    min_memory_used = float('inf')  
    
    for gpu in gpus:  
        # 当前GPU的ID和内存使用情况  
        gpu_id = gpu.id  
        memory_used = gpu.memoryUsed  
        
        # 判断当前GPU是否使用的内存最少  
        if memory_used < min_memory_used:  
            min_memory_used = memory_used  
            least_used_gpu_id = gpu_id  
            
    # 计算剩余可用内存  
    available_memory = gpus[least_used_gpu_id].memoryFree  

    print(f"当前内存使用最少的 GPU 序号: {gpu_id}")  
    print(f"剩余可用内存: {available_memory} MB")

    return least_used_gpu_id 

def gpu_use():
    # 获取可用的GPU列表  
    gpus = GPUtil.getGPUs()  
    
    # 初始化变量来跟踪最少使用的GPU  
    least_used_gpu_id = None  
    min_memory_used = float('inf')  
    
    for gpu in gpus:  
        # 当前GPU的ID和内存使用情况  
        gpu_id = gpu.id  
        memory_free = gpu.memoryFree  
        
        print(f"GPU序号: {gpu_id}")  
        print(f"剩余可用内存: {memory_free} MB\n")

    return least_used_gpu_id 


if __name__ == "__main__":
    # get_least_used_gpu()
    gpu_use()

