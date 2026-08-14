<template>
  <div>
    <el-card class="welcome-card">
      <h2>TensorLBM AI4S Platform</h2>
      <p>HPC 数据生产 → 数据管理 → AI4S 模型开发 → 全栈服务</p>
    </el-card>

    <el-row :gutter="16" class="module-row">
      <el-col :span="6" v-for="m in modules" :key="m.path">
        <el-card class="module-card" @click="$router.push(m.path)">
          <el-icon :size="28"><component :is="m.icon" /></el-icon>
          <h4>{{ m.title }}</h4>
          <p>{{ m.desc }}</p>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
const modules = [
  { path: '/production/cad', icon: 'Box', title: '数据生产', desc: 'CAD → 预处理 → 求解 → 后处理' },
  { path: '/ai4s/apps', icon: 'MagicStick', title: 'AI4S 应用', desc: '9 个 AI4S 应用（FNO/PINN/GNN...）' },
  { path: '/data', icon: 'Coin', title: '数据管理', desc: '数据目录 / 质量 / 血缘' },
  { path: '/misc/projects', icon: 'Folder', title: '项目', desc: '项目管理 / 报告' },
]
</script>

<style scoped>
.welcome-card {
  margin-bottom: 16px;
}
.module-row {
  margin-top: 8px;
}
.module-card {
  cursor: pointer;
  text-align: center;
  transition: all 0.2s;
}
.module-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}
.module-card h4 {
  margin: 8px 0 4px;
}
.module-card p {
  margin: 0;
  font-size: 12px;
  color: #909399;
}
</style>
