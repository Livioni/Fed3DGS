import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# 数据准备
iterations = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
psnr = [19.4417, 19.9075, 20.2999, 20.3186, 20.5901, 20.5392, 20.4626, 20.3473, 20.0884, 19.6449]
ssim = [0.5370, 0.5712, 0.5853, 0.5899, 0.5922, 0.5926, 0.5909, 0.5927, 0.5675, 0.5396]
lpips = [0.5067, 0.4639, 0.4451, 0.4351, 0.4302, 0.4270, 0.4249, 0.4190, 0.4476, 0.4745]

def add_values(ax, x_data, y_data, fmt="{:.4f}"):
    for i, value in enumerate(y_data):
        ax.text(x_data[i], value, fmt.format(value), fontsize=18, ha='left')

# 创建一个图和三个子图
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# 第一个子图 - PSNR
axs[0].plot(iterations, psnr, marker='o', color='b', label='PSNR', linewidth=3)
axs[0].set_title('PSNR Over Iterations',fontsize='22')
axs[0].set_ylabel('PSNR',fontsize='22')
axs[0].grid(True)
axs[0].legend()
add_values(axs[0], iterations, psnr)  # 显示数据点值
axs[0].tick_params(axis='both', which='major', labelsize=20)

# 第二个子图 - SSIM
axs[1].plot(iterations, ssim, marker='s', color='g', label='SSIM',linewidth=3)
axs[1].set_title('SSIM Over Iterations',fontsize='22')
axs[1].set_ylabel('SSIM',fontsize='22')
axs[1].grid(True)
axs[1].legend()
add_values(axs[1], iterations, ssim)  # 显示数据点值
axs[1].tick_params(axis='both', which='major', labelsize=20)

# 第三个子图 - LPIPS
axs[2].plot(iterations, lpips, marker='^', color='r', label='LPIPS',linewidth=3)
axs[2].set_title('LPIPS Over Iterations',fontsize='22')
axs[2].set_xlabel('Iterations',fontsize='22')
axs[2].set_ylabel('LPIPS',fontsize='22')
axs[2].grid(True)
axs[2].legend()
add_values(axs[2], iterations, lpips)  # 显示数据点值
axs[2].tick_params(axis='both', which='major', labelsize=20)


# 展示整个图
# plt.show()
plt.tight_layout()
plt.savefig("visualizations/PDF/metrics.png", dpi=300, bbox_inches='tight')