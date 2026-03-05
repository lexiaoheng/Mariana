import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

from Mariana_Core import NoiseAttenuation, utils

# 读数据
data = scio.loadmat('./test_data/testdata_for_noiseattenuation.mat')['data'].astype('float32')

# 合成含噪数据
[h, w] = data.shape
noisy_data = data + 1.5 * np.random.randn(h, w)
print('SNR of noisy data:', utils.evaluate(data, noisy_data, 'SNR'))

# 使用扩散模型处理
out = NoiseAttenuation.DPM(noisy_data, ddim=True)
print('SNR of processed data:', utils.evaluate(data, out, 'SNR'))

# 展示处理结果
plt.subplot(1, 3, 1)
plt.imshow(data, vmin=-4, vmax=4, cmap='RdGy', aspect='auto')
plt.title('Ground truth')

plt.subplot(1, 3, 2)
plt.imshow(noisy_data, vmin=-4, vmax=4, cmap='RdGy', aspect='auto')
plt.title('Noisy data')

plt.subplot(1, 3, 3)
plt.imshow(out, vmin=-4, vmax=4, cmap='RdGy', aspect='auto')
plt.title('Processed by ddpm')

plt.show()
