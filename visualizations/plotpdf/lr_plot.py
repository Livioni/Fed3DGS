import numpy as np
import matplotlib.pyplot as plt

# 定义学习率函数
def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# 参数设置
lr_init = 0.00016
lr_final = 0.0000016
lr_delay_steps = 15000  # 假设延迟步数是 2000，需要调整来观察效果
lr_delay_mult = 0.5
max_steps = 20000

# 创建学习率函数
lr_function = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps)

# 生成数据点
steps = np.arange(max_steps + 1)
lr_values = [lr_function(step) for step in steps]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, lr_values, label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.yscale('log')  # 对数尺度以更好地观察变化
# plt.show()
plt.savefig('visualizations/PDF/lr_plot.png')
