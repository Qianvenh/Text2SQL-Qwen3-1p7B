import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
models = [
    'baseline',
    'RAG_only_embedding', 
    'RAG_embedding+rerank',
    'qwen3_1p7B_lora_32r',
    'qwen3_1p7B_lora_128r',
    'qwen3_1p7B_full'
]

difficulties = ['Easy', 'Medium', 'Hard', 'Extra']

# 性能数据 (转换为小数)
performance_data = {
    'baseline': [85.9, 58.7, 43.1, 15.1],
    'RAG_only_embedding': [83.9, 57.0, 39.1, 12.7],
    'RAG_embedding+rerank': [83.5, 56.7, 41.4, 16.3],
    'qwen3_1p7B_lora_32r': [79.8, 47.3, 37.4, 19.3],
    'qwen3_1p7B_lora_128r': [73.4, 42.6, 32.2, 15.1],
    'qwen3_1p7B_full': [85.5, 57.6, 52.3, 21.7]
}

# 设置图形大小和样式
fig, ax = plt.subplots(figsize=(12, 8))

# 设置柱状图的宽度和位置
x = np.arange(len(difficulties))
width = 0.13
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

# 绘制每个模型的柱状图
for i, model in enumerate(models):
    offset = (i - 2.5) * width
    bars = ax.bar(x + offset, performance_data[model], width, 
                  label=model, color=colors[i], alpha=0.8)
    
    # 在柱状图顶部添加数值标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# 设置图表属性
ax.set_xlabel('Task Difficulty', fontsize=14, fontweight='bold')
ax.set_ylabel('Execution Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison Across Different Task Difficulties', 
             fontsize=16, fontweight='bold', pad=20)

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(difficulties)

# 设置y轴范围
ax.set_ylim(0, 100)

# 添加网格线
ax.grid(True, alpha=0.3, linestyle='--')

# 设置图例
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('./figs/model_performance_comparison.png', 
            dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

print("图表已生成并保存为 'model_performance_comparison.png'")
